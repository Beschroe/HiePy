"""
HTTensor dimtree restructuring.

This module contains a function that restructures the dimension tree of a
HTTensor such that a specified node becomes the left or right child of the
root.

###############################################################
##### WARNING: This module is currently under maintenance.#####
###############################################################
"""

# DEPRECATED

# from __future__ import annotations

# import torch

# from hiepy.core.httensor.httensor import HTTensor as HTTensor


# def change_root(x: HTTensor, node: tuple[int, ...], lr: str = "right") -> HTTensor:
#     """
#     Hinweis: Dies ist eine interne Funktion der Funktion HTucker.tensordot
#     ______________________________________________________________________
#     Strukturiert den Dimensionsbaum des hierarchischen Tuckertensors 'self' so um, dass der Knoten 'node' linkes
#     oder rechtes (vgl. Parameter 'lr') Kind der Wurzel wird. Ist 'node' selbst die Wurzel, wird eine neue Wurzel
#     eingefuegt. Die Aenderungen werden direkt auf 'self' durchgefuehrt.
#     ______________________________________________________________________
#     Parameter:
#     - node (int,...): Ein integer tuple wie beispielsweise (0,1) oder (2,), das einen Knoten des Dimensionsbaums von
#                       self' repraesentiert.
#     - lr str: Einer der beiden Strings "left" oder "right"
#     Output:
#     None
#     ______________________________________________________________________
#     Beispiel:
#     a) change_root(node=(0,1), lr="right")
#                  (0,1,2,3)                                      (2,3,0,1)
#              (0,1)       (2,3)             ~~~>             (2,3)       (0,1)
#          (0,)    (1,) (2,)   (3,)                        (2,)   (3,) (0,)   (1,)

#     b) change_root(node=(3,4), lr="left")
#                     (0,1,2,3,4,5)                           (3,4,0,1,2,3,5)
#               (0,1,2)           (3,4,5)    ~~~>         (3,4)             (0,1,2,5)
#           (0,1)     (2,)    (3,4)     (5,)           (3,)   (4,)    (0,1,2)     (5,)
#        (0,)   (1,)       (3,)   (4,)                            (0,1)     (2,)
#                                                           (0,)  (1,)

#     c) change_root(node=(0,1,2), lr="right")
#                      (0,1,2)                           (0,1,2,3)
#                  (0,1)     (2,)           ~~~>      (0,)       (1,2,3)
#               (0,)   (1,)                                  (1,2)     (3,)
#                                                         (1,)   (2,)
#     """
#     if not isinstance(node, tuple):
#         raise TypeError("Argument 'node': type(node)={} | node ist kein tuple.".format(type(node)))
#     if not x.dtree.contains(node):
#         raise ValueError("Argument 'node': node={} | node ist kein Knoten des Dimensionsbaums.".format(node))
#     if lr not in ["left", "right"]:
#         raise ValueError("Argument 'lr': lr={} ist weder 'left' noch 'right'.".format(lr))

#     # Aus Lesbarkeitsgruenden
#     x = x
#     dtx = x.dtree

#     # Ist node der Wurzelknoten, muss ein neuer Wurzelknoten erzeugt und eingefuegt werden
#     if dtx.is_root(node):
#         # Fall 1) node ist der Wurzelknoten
#         return _make_root_to_child_of_new_root(x, lr)
#     else:
#         # Fall 2) node ist nicht der Wurzelknoten
#         return _make_node_to_child_of_root(x, node, lr)


# def _make_node_to_child_of_root(x: HTTensor, node: tuple, lr: str = "right") -> HTTensor:
#     """
#     Hinweis: Dies ist eine interne Funktion der Funktion change_root
#     Sei x=self. Modelliert den hierarchischen Tuckertensor x so um, dass der Knoten node linkes (lr="left")
#     oder rechtes (lr="right") Kind des Wurzelknotens ist.
#     """
#     # Aus Lesbarkeitsgruenden
#     dtx = x.dtree
#     root = dtx.get_root()

#     assert dtx.contains(node)
#     assert lr in ["left", "right"]

#     # Fall 1)
#     # node ist bereits Kind der Wurzel. Ggf. soll node aber linkes/rechtes Kind sein und ist
#     # derzeit aber rechtes/linkes Kind
#     if dtx.get_parent(node) == root:
#         node_lr = "left" if dtx.is_left(node) else "right"
#         if lr != node_lr:
#             # Passe Dimensionsbaum entsprechend an
#             new_children = dtx.get_children(root)[::-1]
#             new_root = new_children[0] + new_children[1]
#             dtx.set_children(root, new_children)
#             dtx.replace_node(root, new_root)
#             # Passe Transfertensor der Wurzel entsprechend an
#             x.B[new_root] = torch.transpose(x.B[root], 0, 1)
#             del x.B[root]
#     else:
#         # Fall 2)
#         # node ist noch kein Kind der Wurzel
#         # 1) Entferne Transfertensor der Wurzel
#         # 1.1) Matriziere Transfertensor der Wurzel
#         x.B[root] = torch.squeeze(x.B[root], dim=2)
#         # 1.2) Multipliziere diese Matrix mit rechtem Kind
#         child = dtx.get_right(root)
#         if dtx.is_leaf(child):
#             x.U[child] = torch.tensordot(x.U[child], x.B[root], dims=[[1], [1]])  # type: ignore
#         else:
#             x.B[child] = torch.tensordot(x.B[child], x.B[root], dims=[[2], [1]])  # type: ignore
#         # 1.3) Ersetze nun den Transfertensor der Wurzel durch eine auf 3D geshapte Identitaetsmatrix
#         if node in x.B:
#             x.B[root] = torch.unsqueeze(torch.eye(x.B[node].shape[2]), dim=2)
#         else:
#             x.B[root] = torch.unsqueeze(torch.eye(x.U[node].shape[1]), dim=2)
#         # 2) Setze die neuen Kinder der Wurzel
#         new_nodes = dict(dtx.nodes)
#         if lr == "left":
#             new_nodes[root] = [node, dtx.get_parent(node)]
#         else:
#             new_nodes[root] = [dtx.get_parent(node), node]
#         # 3) Invertiere Knotenhierarchie von node bis zur Wurzel
#         # 3.1) Traversiere Dimensionsbaum von node bis zur Wurzel
#         current_node = node
#         current_node_parent = dtx.get_parent(node)
#         while current_node_parent != root:
#             # current_node ist linkes oder rechtes Kind
#             ch_idx = 0 if dtx.is_left(current_node) else 1
#             # setze current_node auf seinen Vaterknoten
#             current_node = current_node_parent
#             # Berechne neuen Vaterknoten
#             current_node_parent = dtx.get_parent(current_node)
#             # Aktualisiere Kinder von current node i.A. davon, ob der vorherige current_node links oder rechts war
#             current_children = dtx.get_children(current_node)[::1]  # [::1] erzeugt Kopie der Liste
#             if dtx.is_root(current_node_parent):
#                 # 3.2) Haenge das zuvor von der Wurzel getrennte Kind an current_node
#                 current_node_lr = 1 if dtx.is_left(current_node) else 0
#                 # Aktualisiere Kinder
#                 current_children[ch_idx] = dtx.get_children(root)[current_node_lr]  # type: ignore
#                 new_nodes[current_node] = current_children
#                 # Aktualisiere Transfertensor
#                 x.B[current_node] = torch.transpose(x.B[current_node], 2, ch_idx)  # type: ignore
#             else:
#                 # Aktualisiere Kinder von current_node
#                 current_children[ch_idx] = current_node_parent  # type: ignore
#                 new_nodes[current_node] = current_children
#                 # Aktualisiere Transfertensor
#                 x.B[current_node] = torch.transpose(x.B[current_node], 2, ch_idx)  # type: ignore

#         # Berechne nun fuer jeden inneren Knoten neu, welche Dimensionen durch ihn repraesentiert werden
#         old2new = _get_dimensions_for_each_node(new_nodes)  # type: ignore
#         # Benenne die Knoten darauf aufbauend um
#         new_nodes = {old2new[k]: [old2new[child] for child in v] for k, v in new_nodes.items()}
#         x.U = {old2new[k]: v for k, v in x.U.items()}
#         x.B = {old2new[k]: v for k, v in x.B.items()}
#         # Passe die Instanzvariablen des hierarchischen Tuckertensors an
#         x.dtree = type(dtx)(new_nodes)
#     return x


# def _make_root_to_child_of_new_root(x: HTTensor, lr: str = "right") -> HTTensor:
#     """
#     Hinweis: Dies ist eine interne Funktion der Funktion change_root.
#     Fuegt in den Dimensionsbaum von x einen neuen Knoten ein und setzt diesen als Wurzel. Dabei wird der
#     vorherige Wurzelknoten als Kind der neuen Wurzel ausgewaehlt. Ob die alte Wurzel zum rechten oder linken Kind wird,
#     haengt von der Wahl von str ("left" oder "right") ab. Das zweite Kind der neuen Wurzel wird ein neu erzeugter
#     Blattknoten, der eine Singleton Dimension repraesentiert.
#     """
#     assert lr in ["left", "right"]

#     # Aus Gruenden der Lesbarkeit
#     dtx = x.dtree
#     old_root = dtx.get_root()

#     if lr == "right":
#         # Alte Wurzel wird zum rechten Kind einer neuen Wurzel
#         # Inkrementiere die abgebildeten Dimensionen jedes alten Knotens um 1
#         # Bsp: (0,1,2) -> (1,2,3), (0,) -> (1,), etc..
#         old2new = {node: tuple(item + 1 for item in node) for node in dtx.get_nodes()}
#         adapted_nodes = {old2new[k]: [old2new[child] for child in v] for k, v in dtx.nodes.items()}
#         adapted_old_root = old2new[old_root]
#         new_left_child_of_new_root = tuple([0])
#         new_root = new_left_child_of_new_root + adapted_old_root
#         adapted_nodes[new_root] = [new_left_child_of_new_root, adapted_old_root]
#         adapted_nodes[new_left_child_of_new_root] = []
#         # Ersetze den Dimensionsbaum von x durch einen neuen Dimensionsbaum basierend auf der aktualisierten
#         # Knotenhierarchie enthalten in adapted_nodes
#         x.dtree = type(dtx)(nodes=adapted_nodes)
#         # Aktualisiere das Blattmatrixdict und Transfertensordict
#         # 1) Aktualisiere die Keys
#         x.U = {old2new[node]: mat for node, mat in x.U.items()}
#         x.B = {old2new[node]: tens for node, tens in x.B.items()}
#         # 2) Fuege fuer die neue Wurzel und die neue Singleton Dimension entsprechende Eintraege hinzu
#         x.U[new_left_child_of_new_root] = torch.ones(1, 1)
#         x.B[new_root] = torch.ones(1, 1, 1)
#     else:
#         # Alte Wurzel wird zum linken Kind einer neuen Wurzel
#         adapted_nodes = dtx.nodes
#         new_right_child_of_new_root = tuple([max(old_root) + 1])
#         new_root = old_root + new_right_child_of_new_root
#         adapted_nodes[new_root] = [old_root, new_right_child_of_new_root]
#         adapted_nodes[new_right_child_of_new_root] = []
#         # Ersetze den Dimensionsbaum von x durch einen neuen Dimensionsbaum basierend auf der aktualisierten
#         # Knotenhierarchie enthalten in adapted_nodes
#         x.dtree = type(dtx)(nodes=adapted_nodes)
#         # Aktualisiere das Blattmatrixdict und Transfertensordict
#         # Fuege fuer die neue Wurzel und die neue Singleton Dimension entsprechende Eintraege hinzu
#         x.U[new_right_child_of_new_root] = torch.ones(1, 1)
#         x.B[new_root] = torch.ones(1, 1, 1)
#     return x


# def _get_dimensions_for_each_node(
#     nodes: dict[tuple[int, ...], list[tuple[int, ...]]],
# ) -> dict:
#     """
#     Hinweis: Das ist eine interne Funktion von _make_node_to_child_of_root
#     """
#     assert isinstance(nodes, dict)
#     # Berechne die Wurzel aus nodes
#     all_children = [child for children in nodes.values() for child in children]
#     root = [node for node in nodes.keys() if node not in all_children][0]
#     # Berechne darauf aufbauend das Mapping fuer die inneren Knoten
#     old2new = {}
#     _get_dimensions_for_each_node_helper(nodes, root, old2new)
#     return old2new


# def _get_dimensions_for_each_node_helper(
#     nodes: dict[tuple[int, ...], list[tuple[int, ...]]],
#     node: tuple[int, ...],
#     mapping: dict,
# ) -> tuple[tuple[int, ...]] | tuple[int, ...]:
#     """
#     Hinweis: Das ist eine interne Funktion von _make_node_to_child_of_root
#     """
#     if node in mapping:
#         return mapping[node]
#     elif not nodes[node]:
#         mapping[node] = node
#         return node
#     else:
#         l, r = nodes[node]
#         mapping[node] = _get_dimensions_for_each_node_helper(nodes, l, mapping) + _get_dimensions_for_each_node_helper(
#             nodes, r, mapping
#         )
#         return mapping[node]
