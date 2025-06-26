from treelib import Node, Tree
from Biigle.biigle import Api

api = Api()

volume_id =  352
lt_id = 60

maelle_labels = api.get(f'volumes/{volume_id}/statistics').json()[
    'annotationLabels'
]

corentin_labels = api.get(f'volumes/462/statistics').json()[
    'annotationLabels'
]
annotations_labels = maelle_labels + corentin_labels

lt = api.get(f'label-trees/{lt_id}').json()["labels"]


tree = Tree()
tree.create_node("Root", "root")
while len(lt) > 0:
    for l in lt:
        if l["parent_id"] is None:
            tree.create_node(l["name"], l["id"], parent="root")
            lt.remove(l)
        else:
            if tree.contains(l["parent_id"]):
                nb = 0
                for al in annotations_labels:
                    if al["id"] == l["id"]:
                        nb = al["count"]
                        break
                tree.create_node(f"{l['name']}: {nb}", l["id"], parent=l["parent_id"], data = nb)
                lt.remove(l)
                if l["name"] == "Animalia":
                    animalia_id = l["id"]

tree = tree.subtree(animalia_id)

for _ in range(5):
    for node in tree.leaves():
        if node.data is not None:
            if node.data == 0:
                tree.remove_node(node.identifier)


print(tree)
tree.to_graphviz(r"D:test.dot")
print("stop")







