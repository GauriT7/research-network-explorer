
import streamlit as st
import networkx as nx
import streamlit.components.v1 as components
import json
import re
import hashlib
import requests
import fitz
import tempfile
import os
from pyvis.network import Network
from openai import OpenAI
from neo4j import GraphDatabase
from urllib.parse import quote

# ============================================================
# CONFIG
# ============================================================

st.set_page_config(
    page_title="Research Nexus",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.stApp {
    background: #0b1020;
    color: #e8eaf0;
}
[data-testid="stSidebar"] {
    background: #121a2c;
}
h1, h2, h3 {
    color: #dce6ff !important;
}
.stTextInput input, .stTextArea textarea {
    background: #182238;
    color: white;
}
div[data-testid="stMetric"] {
    background: #182238;
    padding: 15px;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# SECRETS / CONNECTIONS
# ============================================================

import os

def get_secret(key, default=""):
    return os.getenv(key) or st.secrets.get(key, default)


OPENAI_API_KEY = get_secret("OPENAI_API_KEY")
NEO4J_URI = get_secret("NEO4J_URI")
NEO4J_USERNAME = get_secret("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = get_secret("NEO4J_PASSWORD")
OPENALEX_EMAIL = get_secret("OPENALEX_EMAIL", "")

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None


# ============================================================
# NEO4J DATABASE
# ============================================================

@st.cache_resource
def get_driver():
    if not NEO4J_URI or not NEO4J_PASSWORD:
        return None

    return GraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4J_USERNAME, NEO4J_PASSWORD)
    )


driver = get_driver()
def check_database_connection():
    if driver is None:
        return False
    try:
        driver.verify_connectivity()
        return True
    except Exception as e:
        st.error(f"Neo4j connection failed: {e}")
        return False


def init_database():
    if driver:
        with driver.session() as session:
            session.run("""
                CREATE CONSTRAINT entity_id IF NOT EXISTS
                FOR (n:Entity)
                REQUIRE n.id IS UNIQUE
            """)


def save_entity(node):
    if not driver:
        return

    with driver.session() as session:
        session.run("""
            MERGE (n:Entity {id: $id})
            SET n.name = $name,
                n.type = $type,
                n.description = $description,
                n.updated = datetime()
            """,
            id=node["id"],
            name=node["name"],
            type=node["type"],
            description=node.get("description", "")
        )


def save_relationship(edge):
    if not driver:
        return

    with driver.session() as session:
        session.run("""
            MATCH (a:Entity {id: $source})
            MATCH (b:Entity {id: $target})
            MERGE (a)-[r:RELATED {type: $relationship}]->(b)
            SET r.source_document = $document,
                r.evidence = $evidence,
                r.updated = datetime()
            """,
            source=edge["source"],
            target=edge["target"],
            relationship=edge["relationship"],
            document=edge.get("document", ""),
            evidence=edge.get("evidence", "")
        )


def load_graph():
    G = nx.Graph()

    if not driver:
        return G

    with driver.session() as session:

        nodes = session.run("""
            MATCH (n:Entity)
            RETURN n.id AS id,
                   n.name AS name,
                   n.type AS type,
                   n.description AS description
        """)

        for record in nodes:
            G.add_node(
                record["id"],
                name=record["name"],
                type=record["type"],
                description=record["description"] or ""
            )

        edges = session.run("""
            MATCH (a:Entity)-[r:RELATED]->(b:Entity)
            RETURN a.id AS source,
                   b.id AS target,
                   r.type AS relationship,
                   r.evidence AS evidence,
                   r.source_document AS document
        """)

        for record in edges:
            G.add_edge(
                record["source"],
                record["target"],
                relationship=record["relationship"],
                evidence=record["evidence"] or "",
                document=record["document"] or ""
            )

    return G


init_database()


# ============================================================
# ENTITY NORMALIZATION
# ============================================================

def normalize_name(name):
    name = name.lower().strip()
    name = re.sub(r"^(dr|prof|professor)\.?\s+", "", name)
    name = re.sub(r"[^\w\s]", "", name)
    name = re.sub(r"\s+", " ", name)
    return name.strip()


def make_entity_id(name, entity_type):
    normalized = normalize_name(name)
    return f"{entity_type.lower()}::{normalized}"


# ============================================================
# GRAPH MANAGEMENT
# ============================================================

if "graph" not in st.session_state:
    st.session_state.graph = load_graph()

if "documents" not in st.session_state:
    st.session_state.documents = []

if "selected_entity" not in st.session_state:
    st.session_state.selected_entity = None

G = st.session_state.graph


def add_entity(name, entity_type, description=""):
    entity_id = make_entity_id(name, entity_type)

    if entity_id in G:
        old = G.nodes[entity_id].get("description", "")
        description = description or old

    G.add_node(
        entity_id,
        name=name,
        type=entity_type,
        description=description
    )

    save_entity({
        "id": entity_id,
        "name": name,
        "type": entity_type,
        "description": description
    })

    return entity_id


def add_relationship(source, target, relationship,
                   evidence="", document=""):

    if source not in G or target not in G:
        return

    G.add_edge(
        source,
        target,
        relationship=relationship,
        evidence=evidence,
        document=document
    )

    save_relationship({
        "source": source,
        "target": target,
        "relationship": relationship,
        "evidence": evidence,
        "document": document
    })


def get_entity_by_name(name):
    normalized = normalize_name(name)

    for node, attrs in G.nodes(data=True):
        if normalize_name(attrs.get("name", node)) == normalized:
            return node

    return None


# ============================================================
# DOCUMENT EXTRACTION
# ============================================================

def extract_pdf_text(file):
    document = fitz.open(stream=file.read(), filetype="pdf")

    pages = []

    for page in document:
        pages.append(page.get_text("text"))

    return "\n".join(pages)


def extract_image_text(file):
    if not client:
        return ""

    import base64

    encoded = base64.b64encode(file.getvalue()).decode("utf-8")

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Extract all readable text from this image. "
                            "Preserve names, institutional affiliations, "
                            "titles, and publication details. "
                            "Return plain text only."
                        )
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{file.type};base64,{encoded}"
                        }
                    }
                ]
            }
        ],
        max_tokens=4000
    )

    return response.choices[0].message.content


# ============================================================
# AI STRUCTURED KNOWLEDGE EXTRACTION
# ============================================================

def extract_knowledge(text, filename):

    if not client:
        st.error("OpenAI API key is missing.")
        return None

    instructions = """
You are a scientific research intelligence extraction system.

Extract entities and relationships explicitly supported by the document.

ENTITY TYPES:
Person
Institution
Organisation
Publication
ResearchField
Conference
ClinicalTrial
Grant

RELATIONSHIPS:
educated_at
employed_at
affiliated_with
holds_chair_at
founded
co_authored
authored
mentored_by
collaborated_with
member_of
presented_at
funded_by
conducts_research_in
cited
participated_in

RULES:
- Extract full names whenever available.
- Distinguish institutions from departments.
- Include past and present affiliations if stated.
- Never invent mentorship or personal relationships.
- Shared affiliation is NOT proof of collaboration.
- Extract only evidence-supported relationships.
- Include a brief evidence excerpt for each relationship.
- Use consistent entity names.
- If the document contains a list of authors, extract all authors.
- Use null or empty values for unknown details.

Return valid JSON in this exact structure:

{
  "nodes": [
    {
      "name": "Full entity name",
      "type": "Person",
      "description": "Brief factual description"
    }
  ],
  "edges": [
    {
      "source": "Entity A",
      "target": "Entity B",
      "relationship": "co_authored",
      "evidence": "Evidence from the document"
    }
  ]
}
"""

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {
                "role": "system",
                "content": instructions
            },
            {
                "role": "user",
                "content": (
                    f"Source document: {filename}\n\n"
                    f"DOCUMENT CONTENT:\n{text[:30000]}"
                )
            }
        ],
        response_format={"type": "json_object"},
        temperature=0
    )

    return json.loads(
        response.choices[0].message.content
    )


# ============================================================
# INGEST AND RECONCILE
# ============================================================

def ingest_knowledge(data, filename):

    name_to_id = {}

    # 1. Create / resolve nodes
    for item in data.get("nodes", []):

        name = item.get("name", "").strip()
        entity_type = item.get("type", "Unknown")

        if not name:
            continue

        existing = get_entity_by_name(name)

        if existing:
            entity_id = existing

            # Preserve existing descriptions when new data is blank.
            old_description = G.nodes[existing].get(
                "description", ""
            )

            description = (
                item.get("description", "")
                or old_description
            )

            G.nodes[existing]["description"] = description

        else:
            entity_id = add_entity(
                name,
                entity_type,
                item.get("description", "")
            )

        name_to_id[name] = entity_id

    # 2. Resolve edges
    for edge in data.get("edges", []):

        source_name = edge.get("source", "")
        target_name = edge.get("target", "")

        source = (
            name_to_id.get(source_name)
            or get_entity_by_name(source_name)
        )

        target = (
            name_to_id.get(target_name)
            or get_entity_by_name(target_name)
        )

        if source and target and source != target:

            add_relationship(
                source,
                target,
                edge.get("relationship", "connected"),
                edge.get("evidence", ""),
                filename
            )


# ============================================================
# EXTERNAL RESEARCH ENRICHMENT
# ============================================================

def search_openalex(name):

    try:
        params = {
            "search": name,
            "per-page": 5
        }

        if OPENALEX_EMAIL:
            params["mailto"] = OPENALEX_EMAIL

        response = requests.get(
            "https://api.openalex.org/authors",
            params=params,
            timeout=20
        )

        response.raise_for_status()

        results = response.json().get("results", [])

        return [
            {
                "name": author.get("display_name"),
                "works_count": author.get("works_count"),
                "cited_by_count": author.get("cited_by_count"),
                "id": author.get("id"),
                "orcid": author.get("orcid"),
                "affiliations": [
                    item.get("institution", {}).get("display_name")
                    for item in author.get("affiliations", [])
                    if item.get("institution")
                ]
            }
            for author in results
        ]

    except Exception as e:
        return {"error": str(e)}


def search_pubmed(name):

    try:
        response = requests.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
            params={
                "db": "pubmed",
                "term": f"{name}[Author]",
                "retmax": 10,
                "retmode": "json"
            },
            timeout=20
        )

        response.raise_for_status()

        return response.json().get(
            "esearchresult", {}
        ).get("idlist", [])

    except Exception as e:
        return {"error": str(e)}


# ============================================================
# GRAPH VISUALIZATION
# ============================================================

def render_graph(graph, focus=None, physics=True):

    net = Network(
        height="760px",
        width="100%",
        bgcolor="#0b1020",
        font_color="#e8eaf0",
        cdn_resources="in_line",
        select_menu=False,
        filter_menu=False
    )

    colors = {
        "Person": "#73b9ff",
        "Institution": "#a29bfe",
        "Organisation": "#fd79a8",
        "Publication": "#55efc4",
        "ResearchField": "#ffeaa7",
        "Conference": "#fab1a0",
        "ClinicalTrial": "#81ecec",
        "Grant": "#fdcb6e",
        "Unknown": "#b2bec3"
    }

    visible_nodes = set(graph.nodes)

    # When focusing, show the selected node and its neighbors.
    if focus and focus in graph:
        visible_nodes = {focus} | set(graph.neighbors(focus))

    for node in visible_nodes:

        attrs = graph.nodes[node]
        entity_type = attrs.get("type", "Unknown")
        name = attrs.get("name", node)

        color = colors.get(entity_type, "#b2bec3")

        if focus == node:
            color = "#ff7675"

        net.add_node(
            node,
            label=name,
            title=(
                f"{name}\n"
                f"Type: {entity_type}\n"
                f"{attrs.get('description', '')}"
            ),
            color=color,
            size=30 if entity_type == "Person" else 20,
            font={"size": 16, "face": "Arial"},
            borderWidth=2
        )

    for source, target, attrs in graph.edges(data=True):

        if source not in visible_nodes or target not in visible_nodes:
            continue

        relationship = attrs.get(
            "relationship", "connected"
        )

        net.add_edge(
            source,
            target,
            title=relationship,
            label=relationship.replace("_", " "),
            color="#52617a",
            width=1.5,
            font={"size": 10, "color": "#aab8cc"}
        )

    net.set_options("""
    var options = {
      "nodes": {
        "shape": "dot",
        "scaling": {"min": 12, "max": 35}
      },
      "edges": {
        "smooth": {"type": "continuous"},
        "font": {
          "size": 10,
          "background": "#0b1020",
          "strokeWidth": 0
        }
      },
      "physics": {
        "enabled": true,
        "barnesHut": {
          "gravitationalConstant": -4500,
          "centralGravity": 0.2,
          "springLength": 180,
          "springConstant": 0.035,
          "damping": 0.12
        },
        "stabilization": {
          "enabled": true,
          "iterations": 250
        }
      },
      "interaction": {
        "hover": true,
        "navigationButtons": true,
        "keyboard": true,
        "multiselect": true,
        "tooltipDelay": 200
      }
    }
    """)

    if not physics:
        net.toggle_physics(False)

    return net.generate_html()


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

st.sidebar.title("🧬 Research Nexus")

page = st.sidebar.radio(
    "Explore",
    [
        "Network",
        "Researcher Profiles",
        "Upload & Ingest",
        "Discover Researchers",
        "Manage Database"
    ]
)

st.sidebar.divider()

st.sidebar.metric("Total Entities", G.number_of_nodes())
st.sidebar.metric("Relationships", G.number_of_edges())

if driver:
    try:
        driver.verify_connectivity()
        st.sidebar.success("Database connected")
    except Exception as e:
        st.sidebar.error("Neo4j connection failed")
        st.sidebar.caption(str(e))
else:
    st.sidebar.warning("Database credentials missing")


# ============================================================
# MAIN: NETWORK
# ============================================================

if page == "Network":

    st.title("🕸️ The Research Universe")

    st.caption(
        "A living map of researchers, institutions, "
        "scientific discoveries, and collaborations."
    )

    c1, c2, c3 = st.columns(3)

    c1.metric("Researchers", sum(
        1 for _, a in G.nodes(data=True)
        if a.get("type") == "Person"
    ))

    c2.metric("Institutions", sum(
        1 for _, a in G.nodes(data=True)
        if a.get("type") in ["Institution", "Organisation"]
    ))

    c3.metric("Publications", sum(
        1 for _, a in G.nodes(data=True)
        if a.get("type") == "Publication"
    ))

    if G.number_of_nodes() == 0:

        st.info(
            "Your knowledge graph is empty. "
            "Start by uploading research papers."
        )

    else:

        search = st.text_input(
            "Search the universe",
            placeholder=(
                "Type a researcher, hospital, university, "
                "publication, or field..."
            )
        )

        matching_nodes = []

        if search:
            matching_nodes = [
                node for node, attrs in G.nodes(data=True)
                if search.lower() in (
                    attrs.get("name", node).lower()
                )
            ]

        selected = None

        if matching_nodes:
            selected = st.selectbox(
                "Select entity to explore",
                matching_nodes,
                format_func=lambda x: G.nodes[x].get("name", x)
            )

        show_physics = st.toggle(
            "Interactive physics",
            value=True
        )

        html = render_graph(
            G,
            focus=selected,
            physics=show_physics
        )

        components.html(
            html,
            height=790,
            scrolling=False
        )

        if selected:

            st.markdown("---")

            attrs = G.nodes[selected]

            st.header(
                attrs.get("name", selected)
            )

            st.write(
                attrs.get("description", "No description yet.")
            )

            st.subheader("Direct connections")

            for neighbor in G.neighbors(selected):

                edge = G.edges[selected, neighbor]

                st.write(
                    f"**{G.nodes[neighbor].get('name', neighbor)}** "
                    f"— {edge.get('relationship', 'connected')}"
                )

            st.subheader("Explore paths")

            targets = [
                node for node in G.nodes
                if node != selected
            ]

            if targets:

                target = st.selectbox(
                    "Find connection to:",
                    targets,
                    format_func=lambda x: G.nodes[x].get("name", x)
                )

                try:

                    path = nx.shortest_path(
                        G,
                        selected,
                        target
                    )

                    for i in range(len(path) - 1):

                        a = path[i]
                        b = path[i + 1]

                        relationship = G.edges[a, b].get(
                            "relationship", "connected"
                        )

                        st.markdown(
                            f"**{G.nodes[a].get('name', a)}** "
                            f"→ *{relationship}* → "
                            f"**{G.nodes[b].get('name', b)}**"
                        )

                except nx.NetworkXNoPath:

                    st.warning("No known path between these entities.")


# ============================================================
# RESEARCHER PROFILES
# ============================================================

elif page == "Researcher Profiles":

    st.title("👤 Researcher Intelligence")

    people = [
        node for node, attrs in G.nodes(data=True)
        if attrs.get("type") == "Person"
    ]

    if not people:

        st.info("Upload research papers to populate researcher profiles.")

    else:

        selected_person = st.selectbox(
            "Select a researcher",
            people,
            format_func=lambda x: G.nodes[x].get("name", x)
        )

        attrs = G.nodes[selected_person]

        st.header(attrs.get("name", selected_person))
        st.write(attrs.get("description", ""))

        connections = list(G.neighbors(selected_person))

        st.metric("Network connections", len(connections))

        st.subheader("Academic network")

        for neighbor in connections:

            relationship = G.edges[
                selected_person, neighbor
            ].get("relationship", "connected")

            neighbor_attrs = G.nodes[neighbor]

            st.markdown(
                f"**{neighbor_attrs.get('name', neighbor)}** "
                f"({neighbor_attrs.get('type', 'Unknown')}) "
                f"— *{relationship}*"
            )

        st.subheader("External profile enrichment")

        if st.button("Search OpenAlex + PubMed"):

            name = attrs.get("name", selected_person)

            with st.spinner("Searching scholarly databases..."):

                openalex = search_openalex(name)
                pubmed = search_pubmed(name)

            st.markdown("### OpenAlex")
            st.json(openalex)

            st.markdown("### PubMed publication IDs")
            st.json(pubmed)


# ============================================================
# UPLOAD AND INGEST
# ============================================================

elif page == "Upload & Ingest":

    st.title("📥 Feed the Research Engine")

    st.markdown(
        "Upload PDFs, screenshots, or images. "
        "AI extracts the knowledge and connects it to your existing graph."
    )

    uploaded_files = st.file_uploader(
        "Select documents",
        type=["pdf", "png", "jpg", "jpeg", "webp"],
        accept_multiple_files=True
    )

    if uploaded_files and st.button("🧠 Extract & integrate"):

        if not client:

            st.error(
                "OpenAI API key is missing. "
                "Configure it in Streamlit Secrets."
            )

        else:

            progress = st.progress(0)

            for i, file in enumerate(uploaded_files):

                with st.spinner(f"Processing {file.name}..."):

                    try:

                        if file.type == "application/pdf":

                            text = extract_pdf_text(file)

                        else:

                            text = extract_image_text(file)

                        if not text.strip():

                            st.warning(
                                f"No readable text in {file.name}"
                            )
                            continue

                        knowledge = extract_knowledge(
                            text,
                            file.name
                        )

                        if knowledge:

                            ingest_knowledge(
                                knowledge,
                                file.name
                            )

                            st.success(
                                f"Integrated {file.name}: "
                                f"{len(knowledge.get('nodes', []))} entities, "
                                f"{len(knowledge.get('edges', []))} relationships"
                            )

                            with st.expander("View extracted information"):

                                st.json(knowledge)

                    except Exception as e:

                        st.error(
                            f"Error processing {file.name}: {e}"
                        )

                progress.progress(
                    (i + 1) / len(uploaded_files)
                )

            st.success("Ingestion complete!")
            st.rerun()

    st.divider()

    st.subheader("Manually add a researcher")

    with st.form("manual_person"):

        person_name = st.text_input("Full name")
        position = st.text_input("Current position")
        affiliation = st.text_input("Institution")
        research = st.text_area("Research interests")

        submitted = st.form_submit_button("Add researcher")

        if submitted and person_name:

            person_id = add_entity(
                person_name,
                "Person",
                f"{position}. Research: {research}"
            )

            if affiliation:

                inst_id = add_entity(
                    affiliation,
                    "Institution"
                )

                add_relationship(
                    person_id,
                    inst_id,
                    "affiliated_with",
                    evidence="Manually entered"
                )

            st.success("Researcher added to your graph!")
            st.rerun()


# ============================================================
# DISCOVER RESEARCHERS
# ============================================================

elif page == "Discover Researchers":

    st.title("🔭 Researcher Discovery")

    st.markdown(
        "Find researchers through OpenAlex scholarly metadata."
    )

    query = st.text_input(
        "Research field, institution, or author",
        placeholder="e.g. reproductive endocrinology, Tata Memorial..."
    )

    if query and st.button("Search OpenAlex"):

        with st.spinner("Searching OpenAlex..."):

            try:

                params = {
                    "search": query,
                    "per-page": 15
                }

                if OPENALEX_EMAIL:
                    params["mailto"] = OPENALEX_EMAIL

                response = requests.get(
                    "https://api.openalex.org/authors",
                    params=params,
                    timeout=20
                )

                response.raise_for_status()

                authors = response.json().get("results", [])

                for author in authors:

                    name = author.get("display_name", "")

                    st.markdown(f"### {name}")

                    st.write(
                        "Works:", author.get("works_count", 0),
                        "| Citations:", author.get("cited_by_count", 0)
                    )

                    st.write("ORCID:", author.get("orcid", "Not listed"))

                    affiliations = [
                        a.get("institution", {}).get("display_name")
                        for a in author.get("affiliations", [])
                        if a.get("institution")
                    ]

                    if affiliations:
                        st.write("Affiliations:", ", ".join(affiliations))

                    if st.button(
                        f"Add {name} to network",
                        key=f"add_{author.get('id')}"
                    ):

                        person_id = add_entity(
                            name,
                            "Person",
                            f"OpenAlex profile: {author.get('id')}"
                        )

                        for affiliation in affiliations:

                            inst_id = add_entity(
                                affiliation,
                                "Institution"
                            )

                            add_relationship(
                                person_id,
                                inst_id,
                                "affiliated_with",
                                evidence="OpenAlex affiliation metadata"
                            )

                        st.success(f"Added {name}!")
                        st.rerun()

                    st.divider()

            except Exception as e:

                st.error(f"OpenAlex error: {e}")


# ============================================================
# DATABASE MANAGEMENT
# ============================================================

elif page == "Manage Database":

    st.title("🗄️ Knowledge Base Management")

    st.subheader("Export your graph")

    if G.number_of_nodes() > 0:

        export_data = nx.node_link_data(G)

        st.download_button(
            "Download JSON backup",
            json.dumps(export_data, indent=2),
            "research_nexus_backup.json",
            "application/json"
        )

        node_rows = []

        for node, attrs in G.nodes(data=True):

            node_rows.append({
                "ID": node,
                "Name": attrs.get("name", node),
                "Type": attrs.get("type", ""),
                "Description": attrs.get("description", "")
            })

        st.download_button(
            "Download entities CSV",
            __import__("pandas").DataFrame(
                node_rows
            ).to_csv(index=False),
            "research_entities.csv"
        )

    st.divider()

    st.subheader("Import a previous graph")

    backup = st.file_uploader(
        "Upload JSON backup",
        type=["json"]
    )

    if backup and st.button("Import graph"):

        data = json.load(backup)

        imported = nx.node_link_graph(
            data,
            edges="links"
        )

        for node, attrs in imported.nodes(data=True):

            add_entity(
                attrs.get("name", node),
                attrs.get("type", "Unknown"),
                attrs.get("description", "")
            )

        for source, target, attrs in imported.edges(data=True):

            source_name = imported.nodes[source].get(
                "name", source
            )

            target_name = imported.nodes[target].get(
                "name", target
            )

            a = get_entity_by_name(source_name)
            b = get_entity_by_name(target_name)

            if a and b:

                add_relationship(
                    a,
                    b,
                    attrs.get("relationship", "connected"),
                    attrs.get("evidence", ""),
                    attrs.get("document", "Imported backup")
                )

        st.success("Import complete!")
        st.rerun()

    st.divider()

    st.subheader("Danger zone")

    st.warning(
        "Deleting the graph removes your persistent database data."
    )

    confirm_delete = st.checkbox(
        "I understand and want to clear the database"
    )

    if confirm_delete and st.button("Delete all graph data"):

        if driver:

            with driver.session() as session:
                session.run("""
                    MATCH (n:Entity)
                    DETACH DELETE n
                """)

        st.session_state.graph = nx.Graph()
        st.success("Database cleared.")
        st.rerun()


# ============================================================
# FOOTER
# ============================================================

st.sidebar.caption(
    "Research Nexus · Built for curiosity."
)
