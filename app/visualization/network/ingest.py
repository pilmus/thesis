import itertools

import pandas as pd
from conf.config import driver
from tqdm import tqdm

from network.create_coocc_matrix import create_co_occurences_matrix

ALLOW_LIST = {
    "PFAS",
    "Nederland",
    "PFOS",
    "DPG",
    "Zwijndrecht",
    "Vlaanderen",
    "Europa",
    "Antwerpen",
    "Lantis",
    "Tweede Kamer",
    "Westerschelde",
    "N-VA",
    "Duitsland",
    "Den Haag",
    "RIVM",
    "Amsterdam",
    "Kamer",
    "Europese Commissie",
    "Oosterweelverbinding",
    "Schelde",
    "België",
    "Vlaamse regering",
    "Groningen",
    "Dordrecht",
    "Limburg",
    "Chemours",
    "Groen",
    "Zeeland",
    "OVAM",
    "EU",
    "Hedwigepolder",
    "Vlaams Parlement",
    "PFOA",
    "Noordzee",
    "China",
    "Raad van State",
    "Europese Unie",
    "Vlaamse overheid",
    "Noord-Holland",
    "RWS",
    "PBL",
    "Utrecht",
    "Denemarken",
    "Karl Vrancken",
    "Oosterweel",
    "DuPont",
    "D66",
    "September",
    "Rotterdam",
    "VNG",
    "PZC",
    "VVD",
    "Zweden",
    "TNO",
    "Mechelen",
    "Brussel",
    "Linkeroever",
    "Verenigde Staten",
    "BAM",
    "Maas",
    "Noorwegen",
    "EFSA",
    "Zuid-Holland",
    "CBS",
    "Randstad",
    "EU",
    "provincie Zeeland",
    "Gelderland",
    "Ministerie van Infrastructuur en Waterstaat",
    "VS",
    "ProRail",
    "Eindhoven",
    "ILT",
    "Willebroek",
    "KU Leuven",
    "Drenthe",
    "Italië",
    "Deltares",
    "Frankrijk",
    "Parijs",
    "IPO",
    "August",
    "Waal",
    "Rijkswaterstaat",
    "CD&V",
    "provincie Zuid-Holland",
    "Morgen",
    "Universiteit Antwerpen",
    "Noord-Brabant",
    "Europees Parlement",
    "LNV",
    "KRW",
    "Spanje",
    "CDA",
    "Brabant",
    "Arcadis",
    "Twente",
    "Vlaamse Milieumaatschappij",
    "FBSA",
}


def add_connection(tx, first, first_type, second, second_type, weight):
    # query = f"MATCH (a:{first_type} {{name: '{first}'}}), (b:{second_type} {{name: '{second}'}}) MERGE  (a)-[" \
    #         f":LINK]->(b)"
    #
    # query = "MATCH (a:Person {name: 'Mike'}), (b:Person {name: 'Ike'}) MERGE (a)-[:LINK]->(b)"
    # query= f"MERGE (a:{first_type} {{name:'{first}'}})-[l:LINK]->(b:{second_type} {{name:'{second}'}})"
    # query = (
    #     f"MERGE (a:{first_type} {{name:'{first}'}}) MERGE (b:{second_type} {{name:'{second}'}}) MERGE (a)-["
    #     f"l:LINK]->(b) ON CREATE SET l.count = 1, l.vis = none ON MATCH SET l.count = l.count + 1"
    # )
    # query = f"MERGE (a:{first_type} {{name:'{first}'}}) MERGE (b:{second_type} {{name:'{second}'}}) MERGE (a)-[" \
    #         f"l:LINK]->(b) ON CREATE SET l.weight = 1 ON MATCH SET l.weight = l.weight + 1"
    query = (
        f"MATCH (a:{first_type} {{name:'{first}'}}), (b:{second_type} {{name:'{second}'}}) CREATE (a)-[l:LINK {{"
        f"weight:{weight}}}]->(b)"
    )
    # print(query)
    tx.run(
        # f"MERGE (a:{first_type} {{name: $first}}) MERGE (a)-[:LINK]->(mention:{second_type} {{name: $second}})",
        query
        # first=first,
        # second=second,
    )


def add_node(tx, name, type):
    # tx.run("MERGE (a:$type {name: $name})", name=name, type=type)
    q = f"MERGE (a:{type} {{name: '{name}'}})"
    tx.run(q)


def print_connections(tx, name):
    for record in tx.run(
        "MATCH (a:Entity)-[:LINK]-(entity) WHERE a.name = $name " "RETURN entity.name ORDER BY entity.name", name=name
    ):
        print(record["entity.name"])


def truncate(tx):
    tx.run("match (a) -[r] -> () delete a, r")
    tx.run("match (a) delete a")
    tx.run("MATCH (n) DETACH DELETE n")


def setup(tx):
    tx.run("CREATE DATABASE neo4j")


df = pd.read_pickle("processed-with-entities.pickle")
print(df)
entities = df.entities.to_list()
entities = [[item for item in list_ if item[0] in ALLOW_LIST] for list_ in entities]
# entities = [entities[1], entities[1], entities[3], entities[5]]

unique_entities = list(set(itertools.chain(*entities)))
#
#
# links = []

words_cooc_matrix, word_to_id = create_co_occurences_matrix(unique_entities, entities)
print(type(words_cooc_matrix.todense()))

print(words_cooc_matrix.todense().item((0, 0)))

with driver.session() as session:
    session.write_transaction(truncate)
    # session.write_transaction(setup)
    # session.write_transaction(single_line)
    # session.write_transaction(single_line)

    # entities = entities +  entities[0]+ entities[0]

    # entities = entities + [entities[0][:2]]
    # print(entities)

    for entity in tqdm(unique_entities):
        session.write_transaction(add_node, entity[0], entity[1])

    for i in tqdm(range(0, len(unique_entities))):
        for j in range(i + 1, len(unique_entities)):
            first = unique_entities[i]
            second = unique_entities[j]

            first_id = word_to_id[first]
            second_id = word_to_id[second]

            weight = words_cooc_matrix.todense().item((first_id, second_id))

            # print(first, second)
            # print(first[0], first[1])
            # print(second[0], second[1])
            # session.write_transaction(add_node, list_[i][0], list_[i][1])
            if weight > 0:
                session.write_transaction(add_connection, first[0], first[1], second[0], second[1], weight)

    # for list_ in tqdm(entities):
    #     # print(list_)
    #     list_ = sorted(list_)
    #     for i in range(0, len(list_)):
    #         for j in range(i + 1, len(list_)):
    #             first = list_[i]
    #             second = list_[j]
    #             # print(first, second)
    #             # print(first[0], first[1])
    #             # print(second[0], second[1])
    #             # session.write_transaction(add_node, list_[i][0], list_[i][1])
    #             session.write_transaction(add_connection, first[0], first[1], second[0], second[1])
    # session.write_transaction(set_visibility)
    # session.write_transaction(add_connection, "Arthur", "Lancelot")s
    # session.write_transaction(add_connection, "Arthur", "Merlin")
    # session.read_transaction(print_connections, "Arthur")
#
# driver.close()