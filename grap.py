from neo4j import GraphDatabase

# 🔑 Replace these with your Neo4j Aura credentials
URI = "neo4j+s://537b6216.databases.neo4j.io"
USERNAME = "neo4j"
PASSWORD = "neuAwtw9tu_SOgwpiVGFLGbc8l4b7vgKm7I1ze5e4dI"

# Initialize driver
driver = GraphDatabase.driver(URI, auth=(USERNAME, PASSWORD))

# Function to populate graph with a simple example
def create_graph(tx):
    tx.run("""
        CREATE (p:Person {name: "Barack Obama"})
        CREATE (l:Location {name: "Hawaii"})
        CREATE (p)-[:BORN_IN]->(l)
    """)

# Run the function
with driver.session() as session:
    session.execute_write(create_graph)

print("✅ Graph populated!")

# Close the driver
driver.close()
