from neo4j import GraphDatabase


class Neo4jStore:
    def __init__(self, db_name: str, uri: str = "neo4j://127.0.0.1:7687", user: str = "neo4j",
                 password: str = "12345678"):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.db_name = db_name

    def close(self):
        self.driver.close()

    # --- 写入接口：增强健壮性 ---
    def ingest_chunk(self, chunk_id: str, text: str, entities: list = None, images: list = None):
        """
        创建图节点与关系。
        使用 FOREACH 处理空列表，确保在没有图片或实体时 query 依然能执行成功。
        """
        entities = entities or []
        images = images or []

        # 使用 FOREACH 代替 UNWIND 可以防止空列表导致后续逻辑跳过
        query = """
        MERGE (c:Chunk {id: $cid}) SET c.text = $text
        WITH c
        FOREACH (ent IN $ents | 
            MERGE (e:Entity {name: ent.name}) 
            ON CREATE SET e.type = ent.type 
            MERGE (e)-[:MENTIONED_IN]->(c)
        )
        WITH c
        FOREACH (img IN $imgs | 
            MERGE (i:Image {id: img.id}) 
            SET i.path = img.path
            MERGE (c)-[:HAS_IMAGE]->(i)
        )
        """
        with self.driver.session(database=self.db_name) as session:
            session.run(query, cid=chunk_id, text=text, ents=entities, imgs=images)

    def execute_read_query(self, cypher: str, parameters: dict = None):
        """只读查询接口"""
        with self.driver.session(database=self.db_name) as session:
            return session.execute_read(lambda tx: tx.run(cypher, parameters).data())

    def clear_db(self):
        with self.driver.session(database=self.db_name) as session:
            session.run("MATCH (n) DETACH DELETE n")


def main():
    store = Neo4jStore(db_name="neo4j")  # 对应截图中的运行数据库
    store.clear_db()

    # 1. 原始数据
    raw_1 = [10000,
             "The National Museum of the American Indian in Washington, D.C., opened in 2004 on the Mall, embodies a harmonious fusion of cultural respect and architectural beauty. Its curvilinear structure, adorned with a smooth, beige-colored exterior, exudes an organic and natural feel that resonates with the landscapes it reflects. The building's flowing design draws inspiration from natural forms, blending seamlessly with its surroundings. Under a warm-hued sky, the museum is elegantly illuminated, with soft lighting accentuating the textures and contours of its facade. This institution stands as a testament to the cultural significance of American Indians, who are comfortable with terms like Indian, American Indian, and Native American\u2014reflected in the museum's name.<PIC>",
             [30321533]]
    raw_2 = [10001,
             "The Xanadu House in Kissimmee, Florida, built in 1985, showcases a unique and futuristic architectural design with a distinctive beige exterior. Known for its bulbous, organic shapes that seamlessly blend with the surrounding natural elements, the house reflects in a serene pond and is enveloped by lush trees, creating a harmonious integration of nature and innovation. This architectural marvel was ahead of its time, incorporating an automated system managed by Commodore microcomputers. Within its fifteen rooms, spaces like the kitchen, party room, health spa, and bedrooms were heavily equipped with computers and electronic gadgets, emphasizing advanced technology in their design.<PIC>",
             [30278153]]

    # 2. 模拟固定返回值并拼接路径
    def format_imgs(img_ids):
        # 拼接图片路径：MRAMG-Bench/IMAGE/images/WEB/ID.jpg
        return [{"id": i, "path": f"MRAMG-Bench/IMAGE/images/WEB/{i}.jpg"} for i in img_ids]

    chunk_1 = {"id": "10000_0", "text": raw_1[1], "imgs": format_imgs(raw_1[2])}
    ents_1 = [
        {"name": "National Museum of the American Indian", "type": "Building"},
        {"name": "Beige", "type": "Color"}
    ]

    chunk_2 = {"id": "10001_0", "text": raw_2[1], "imgs": format_imgs(raw_2[2])}
    ents_2 = [
        {"name": "Xanadu House", "type": "Building"},
        {"name": "Beige", "type": "Color"}
    ]

    # 3. 健壮性测试：存入一条没有图片和实体的空数据
    store.ingest_chunk("empty_test", "No content here", [], [])

    # 4. 正式建图
    store.ingest_chunk(chunk_1["id"], chunk_1["text"], ents_1, chunk_1["imgs"])
    store.ingest_chunk(chunk_2["id"], chunk_2["text"], ents_2, chunk_2["imgs"])

    # 5. 图扩展查询 (Expand Cypher)
    expand_cypher = """
    MATCH (e1:Entity {name: "National Museum of the American Indian"})-[:MENTIONED_IN]->(c1:Chunk)
    MATCH (e2:Entity {name: "Xanadu House"})-[:MENTIONED_IN]->(c2:Chunk)
    MATCH (c1)<-[:MENTIONED_IN]-(shared:Entity {type: "Color"})-[:MENTIONED_IN]->(c2)
    OPTIONAL MATCH (c1)-[:HAS_IMAGE]->(i1:Image)
    OPTIONAL MATCH (c2)-[:HAS_IMAGE]->(i2:Image)
    RETURN 
        shared.name AS Color, 
        c1.text AS Text1, i1.path AS Path1,
        c2.text AS Text2, i2.path AS Path2
    """

    results = store.execute_read_query(expand_cypher)

    if results:
        r = results[0]
        print(f"[*] 发现共同颜色: {r['Color']}")
        print(f"[*] 图片1路径: {r['Path1']}")
        print(f"[*] 图片2路径: {r['Path2']}")

    store.close()


if __name__ == "__main__":
    main()
