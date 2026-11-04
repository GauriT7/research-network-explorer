
import streamlit as st
import networkx as nx
import streamlit.components.v1 as components
import pandas as pd
import json
import os
from pyvis.network import Network
from pypdf import PdfReader
from openai import OpenAI

# ─────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────

st.set_page_config(
    page_title="Research Nexus",
    page_icon="🕸️",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background-color: #0b1020;
    color: #e8eaf0;
}
[data-testid="stSidebar"] {
    background-color: #121a2c;
}
h1, h2, h3 {
    color: #dce6ff !important;
}
.stTextInput input, .stTextArea textarea {
    background-color: #182238;
    color: #ffffff;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# SESSION DATABASE
# ─────────────────────────────────────────

if "graph" not in st.session_state:
    st.session_state.graph = nx.Graph()

if "documents" not in st.session_state:
    st.session_state.documents = []

G = st.session_state.graph

# ─────────────────────────────────────────
# AI EXTRACTION
# ─────────────────────────────────────────

def extract_entities(text):
    api_key = st.session_state.get("api_key", "")

    if not api_key:
        return None

    client = OpenAI(api_key=api_key)

    prompt = f"""
Analyze the following research document.

Extract:
1. People
2. Universities
3. Hospitals
4. Research institutes
5. Organisations
6. Publications
7. Research fields
8. Relationships between entities

Return ONLY valid JSON in this format:

{{
  "nodes": [
    {{
      "id": "unique entity name",
      "type": "Person | Institution | Publication | ResearchField",
      "description": "brief description"
    }}
  ],
  "edges": [
    {{
      "source": "entity A",
      "target": "entity B",
      "relationship": "educated_at | employed_at | authored | affiliated_with | researches"
    }}
  ]
}}

Document:
{text[:18000]}
"""

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )

    return json.loads(response.choices[0].message.content)


def add_to_graph(data):
    for node in data.get("nodes", []):
        G.add_node(
            node["id"],
            type=node.get("type", "Unknown"),
            description=node.get("description", "")
        )

    for edge in data.get("edges", []):
        source = edge["source"]
        target = edge["target"]

        if source not in G:
            G.add_node(source, type="Unknown")

        if target not in G:
            G.add_node(target, type="Unknown")

        G.add_edge(
            source,
            target,
            relationship=edge.get("relationship", "connected")
        )


# ─────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────

st.sidebar.title("🕸️ Research Nexus")

page = st.sidebar.radio(
    "Navigate",
    ["Network Explorer", "Add Research", "People Directory", "Settings"]
)

st.sidebar.markdown("---")
st.sidebar.metric("Entities", G.number_of_nodes())
st.sidebar.metric("Connections", G.number_of_edges())

# ─────────────────────────────────────────
# NETWORK EXPLORER
# ─────────────────────────────────────────

if page == "Network Explorer":

    st.title("Research Network")
    st.caption(
        "Explore the interconnected world of researchers, "
        "institutions, and scientific discoveries."
    )

    if G.number_of_nodes() == 0:
        st.info(
            "Your network is empty. Upload a research paper "
            "or add researchers to begin building your graph."
        )

    else:
        search = st.text_input(
            "🔍 Find a person, institution, or research topic"
        )

        selected_node = None

        if search:
            matches = [
                node for node in G.nodes
                if search.lower() in node.lower()
            ]

            if matches:
                selected_node = st.selectbox(
                    "Select a matching entity",
                    matches
                )

        # Graph controls
        col1, col2 = st.columns(2)

        with col1:
            physics = st.toggle("Enable network physics", True)

        with col2:
            show_labels = st.toggle("Show all labels", True)

        net = Network(
            height="720px",
            width="100%",
            bgcolor="#0b1020",
            font_color="#e8eaf0",
            notebook=False,
            directed=False
        )

        net.barnes_hut(
            gravity=-3000,
            central_gravity=0.3,
            spring_length=160,
            spring_strength=0.04,
            damping=0.09
        )

        color_map = {
            "Person": "#74b9ff",
            "Institution": "#a29bfe",
            "Publication": "#55efc4",
            "ResearchField": "#ffeaa7",
            "Organisation": "#fd79a8",
            "Unknown": "#b2bec3"
        }

        for node, attrs in G.nodes(data=True):

            node_type = attrs.get("type", "Unknown")
            node_color = color_map.get(node_type, "#b2bec3")

            if selected_node == node:
                node_color = "#ff7675"

            net.add_node(
                node,
                label=node if show_labels else "",
                title=attrs.get("description", node),
                color=node_color,
                size=25 if node_type == "Person" else 18,
                shape="dot"
            )

        for source, target, attrs in G.edges(data=True):

            relationship = attrs.get("relationship", "connected")

            net.add_edge(
                source,
                target,
                title=relationship,
                label=relationship,
                color="#52617a",
                font={"size": 10, "color": "#aab8cc"},
                width=1.5
            )

        net.set_options("""
        var options = {
          "nodes": {
            "font": {"size": 16, "face": "Inter"},
            "borderWidth": 1
          },
          "edges": {
            "smooth": {"type": "dynamic"},
            "color": {"inherit": false}
          },
          "physics": {
            "enabled": true,
            "barnesHut": {
              "gravitationalConstant": -3000,
              "centralGravity": 0.3,
              "springLength": 160,
              "springConstant": 0.04,
              "damping": 0.09
            },
            "stabilization": {"iterations": 200}
          },
          "interaction": {
            "hover": true,
            "navigationButtons": true,
            "keyboard": true,
            "multiselect": true
          }
        }
        """)

        html = net.generate_html()

        components.html(
            html,
            height=740,
            scrolling=True
        )

        if selected_node:
            st.subheader(f"🔎 {selected_node}")

            attrs = G.nodes[selected_node]

            st.write(
                attrs.get(
                    "description",
                    "No description available yet."
                )
            )

            st.markdown("**Direct connections**")

            neighbors = list(G.neighbors(selected_node))

            for neighbor in neighbors:
                relationship = G.edges[
                    selected_node, neighbor
                ].get("relationship", "connected")

                st.write(f"↔ {neighbor} — *{relationship}*")

            if len(neighbors) >= 2:
                st.markdown("**Connections between your selected entity and others**")

                target = st.selectbox(
                    "Explore a path to another entity",
                    [n for n in G.nodes if n != selected_node]
                )

                try:
                    path = nx.shortest_path(
                        G,
                        source=selected_node,
                        target=target
                    )

                    st.success(
                        " → ".join(path)
                    )

                except nx.NetworkXNoPath:
                    st.warning(
                        "No known connection between these entities yet."
                    )

# ─────────────────────────────────────────
# ADD RESEARCH
# ─────────────────────────────────────────

elif page == "Add Research":

    st.title("📚 Add to your knowledge base")

    api_key = st.text_input(
        "OpenAI API key",
        type="password",
        value=st.session_state.get("api_key", "")
    )

    if api_key:
        st.session_state.api_key = api_key

    st.markdown("### Upload a research paper")

    uploaded_file = st.file_uploader(
        "PDF document",
        type=["pdf"]
    )

    if uploaded_file and st.button("Extract & map connections"):

        with st.spinner("Reading document and extracting entities..."):

            reader = PdfReader(uploaded_file)

            text = "\n".join(
                page.extract_text() or ""
                for page in reader.pages
            )

            result = extract_entities(text)

            if result:
                add_to_graph(result)

                st.session_state.documents.append(
                    uploaded_file.name
                )

                st.success(
                    f"Added {len(result['nodes'])} entities "
                    f"and {len(result['edges'])} connections!"
                )

                st.json(result)

            else:
                st.error(
                    "Please add your OpenAI API key in Settings."
                )

    st.markdown("---")
    st.markdown("### Add a person or institution manually")

    with st.form("manual_entry"):

        entity_name = st.text_input("Name")
        entity_type = st.selectbox(
            "Entity type",
            [
                "Person",
                "Institution",
                "Organisation",
                "Publication",
                "ResearchField"
            ]
        )

        description = st.text_area(
            "Description, position, research interests..."
        )

        submitted = st.form_submit_button(
            "Add to network"
        )

        if submitted and entity_name:

            G.add_node(
                entity_name,
                type=entity_type,
                description=description
            )

            st.success(f"Added {entity_name} to the graph!")

    st.markdown("### Connect two entities")

    with st.form("connect_entities"):

        nodes = list(G.nodes)

        if len(nodes) >= 2:

            source = st.selectbox(
                "First entity",
                nodes
            )

            target = st.selectbox(
                "Second entity",
                [n for n in nodes if n != source]
            )

            relationship = st.selectbox(
                "Relationship",
                [
                    "co_authored",
                    "educated_at",
                    "employed_at",
                    "mentored_by",
                    "collaborated_with",
                    "affiliated_with",
                    "holds_chair_at",
                    "researched"
                ]
            )

            connect = st.form_submit_button(
                "Connect entities"
            )

            if connect:

                G.add_edge(
                    source,
                    target,
                    relationship=relationship
                )

                st.success("Connection added!")

        else:
            st.caption("Add at least two entities first.")

# ─────────────────────────────────────────
# PEOPLE DIRECTORY
# ─────────────────────────────────────────

elif page == "People Directory":

    st.title("👥 People & Organisations")

    if G.number_of_nodes() > 0:

        df = pd.DataFrame([
            {
                "Name": node,
                "Type": attrs.get("type", "Unknown"),
                "Description": attrs.get("description", ""),
                "Connections": G.degree(node)
            }
            for node, attrs in G.nodes(data=True)
        ])

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True
        )

        csv = df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "Download directory as CSV",
            csv,
            "research_directory.csv",
            "text/csv"
        )

    else:
        st.info("Your directory is empty.")

# ─────────────────────────────────────────
# SETTINGS
# ─────────────────────────────────────────

elif page == "Settings":

    st.title("⚙️ Settings")

    st.markdown("### Export your entire network")

    if G.number_of_nodes() > 0:

        data = nx.node_link_data(G)

        st.download_button(
            "Download graph as JSON",
            json.dumps(data, indent=2),
            "research_network.json",
            "application/json"
        )

    st.markdown("### Import an existing network")

    uploaded_json = st.file_uploader(
        "Upload JSON network",
        type=["json"]
    )

    if uploaded_json and st.button("Import network"):

        data = json.load(uploaded_json)

        imported = nx.node_link_graph(
            data,
            edges="links"
        )

        st.session_state.graph = imported

        st.success("Network imported successfully!")

    st.caption(
        "Research Nexus · Personal Research Intelligence"
    )
