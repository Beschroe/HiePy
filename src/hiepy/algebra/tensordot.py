"""HTTensor tensordot.

Contains a function to compute HTTensor contractions along specified axes.


###############################################################
##### WARNING: This module is currently under maintenance.#####
###############################################################
"""

# from __future__ import annotations

# from copy import deepcopy

# import torch

# from hiepy.core.httensor.httensor import HTTensor as HTTensor
# from hiepy.core.utils.dimtree import DimTree


# def tensordot(x: HTTensor, y: HTTensor, dims: list[int] | list[list[int]]) -> HTTensor:
#     """Contraction of two tensors (HTTensor) alogn specified dimensions.

#     Berechnet die Tensorkontraktion der beiden hierarchischen Tuckertensoren 'self' und 'y' entlang der in 'dims'
#     definierten Dimensionen.
#     Hinweis: Die Kontraktion zweier Tensoren 'x' und 'y'. im hierarchischen Tuckerformat ist nur in folgenden Faellen
#     moeglich:
#     1. a) In 'x' gibt es einen Knoten, der entweder genau die Dimensionen, die zu kontrahieren sind oder genau die
#           Dimensionen, die nicht zu kontrahieren sind, enthaelt. Das Gleiche gilt fuer 'y'.
#     ODER
#     1. b) Es sind alle Dimensionen von 'x' zu kontrahieren. Ferner gibt es in 'y' zwei Knoten, deren Vereinigung
#           entweder genau den zu kontrahierenden Dimensionen oder genau den nicht zu kontrahierenden Dimensionen
#           entspricht. Fuer diese beiden Knoten gilt entweder, dass ihre beiden Elternknoten Kinder der Wurzel sind oder
#           dass der Elternknoten des Einen Geschwisterknoten des Anderen ist. Die Rollen von 'x' und 'y' sind
#           austauschbar.
#     UND
#     2) Die zu kontrahierenden Abschnitte der Dimensionsbaeume sind kompatibel.
#     ______________________________________________________________________
#     Parameter:
#     - y HTucker.HTTensor: Der mit 'self' zu kontrahierende hierarchische Tuckertensor.
#     - dims [[int,...], [int,...]]: Die zu kontrahierenden Dimensionen von 'self' (dims[0]) und 'y' (dims[1]).
#     ______________________________________________________________________
#     Output:
#     (HTucker.HTTensor,): Der resultierende kontrahierte hierarchische Tuckertensor.
#     ______________________________________________________________________
#     Beispiel:
#                       HTucker.HTTensor         <~~~>          torch.Tensor
#     a)
#        x = HTTensor.randn((3,4,5,6))             |           x = torch.randn(3,4,5,6)
#        y = HTTensor.randn((4,8))                 |           y = torch.randn(4,8)
#        prod = x.tensordot(y, [[1], [0]])         |           prod = torch.tensordot(x, y, [[1], [0]])
#        prod.shape    # = (3,5,6,8)               |           prod.shape    # = torch.size([3,5,6,8])

#     b)
#        x = HTTensor.randn((3,4,5,6))             |           x = torch.randn(3,4,5,6)
#        y = HTTensor.randn((4,6))                 |           y = torch.randn(4,6)
#        prod = x.tensordot(y, [[1,3], [0,1]])     |           prod = torch.tensordot(x, y, [[1,3], [0,1]])
#        prod.shape    # = (3,5)                   |           prod.shape    # = torch.size([3,5])

#     c)
#        x = HTTensor.randn((3,4,5,6))             |           x = torch.randn(3,4,5,6)
#        y = HTTensor.randn((2,5,7,3,6))           |           y = torch.randn(2,5,7,3,6)
#        prod = x.tensordot(y, [[0,2,3], [3,1,4]]) |           prod = torch.tensordot(x, y, [[0,2,3], [3,1,4]])
#        prod.shape    # = (4,2,7)                 |           prod.shape    # = torch.size([4,2,7])

#     """
#     # Argumentchecks
#     if not isinstance(y, type(x)):
#         raise TypeError("Argument 'y': type(y)={} | y ist kein hierarchischer Tuckertensor.".format(type(y)))
#     if dims is None:
#         dims = [[], []]
#     if not isinstance(dims, list):
#         raise TypeError("Argument 'dims': type(dims)={} | dims ist keine list.".format(type(dims)))
#     if len(dims) != 2:
#         raise ValueError("Argument 'dims': len(dims)={} | dims ist keine list mit zwei Elementen.".format(len(dims)))
#     if not isinstance(dims[0], list) or not isinstance(dims[1], list):
#         raise TypeError("Argument 'dims': dims ist keine list bestehend aus zwei sublists.")
#     if len(dims[0]) != len(dims[1]):
#         raise ValueError("Argument 'dims': die beiden sublists aus dims haben nicht die selbe Laenge.")
#     if not all(0 <= dim < len(x.shape) for dim in dims[0]):
#         raise ValueError("Argument 'dims': dims[0] enthaelt ungueltige Dimensionen.")
#     if not all(0 <= dim < len(y.shape) for dim in dims[1]):
#         raise ValueError("Argument 'dims': dims[1] enthaelt ungueltige Dimensionen.")
#     if len(set(dims[0])) != len(dims[0]):
#         raise ValueError("Argument 'dims': dims[0] enthaelt Duplikate.")
#     if len(set(dims[1])) != len(dims[1]):
#         raise ValueError("Argument 'dims': dims[1] enthaelt Duplikate.")
#     if [x.shape[dim] for dim in dims[0]] != [y.shape[dim] for dim in dims[1]]:
#         raise ValueError(
#             "Argument 'y', 'dims': Die Groessen der zu kontrahierenden Dimensionen fuer x und y"
#             "gegeben durch {} und {} sind nicht kompatibel.".format(
#                 [x.shape[dim] for dim in dims[0]], [y.shape[dim] for dim in dims[1]]
#             )
#         )

#     # Kopiere self und y
#     x = deepcopy(x)
#     y = deepcopy(y)

#     # Aus Lesbarkeitsgruenden
#     dims_x = dims[0]
#     dims_y = dims[1]
#     dtx = x.dtree
#     dty = y.dtree

#     # Berechne die Wurzeln jener minimaler Subtrees, die exakt die Dimensionen aus dimsx oder exakt nicht die
#     # Dimensionen aus dimsx enthalten. Das gleiche fuer y
#     compl_dims_x = sorted(set(range(x.order)) - set(dims_x))
#     roots_x = dtx.get_minimal_nodes_covering_dims(dims_x)
#     compl_roots_x = dtx.get_minimal_nodes_covering_dims(compl_dims_x)
#     compl_dims_y = sorted(set(range(y.order)) - set(dims_y))
#     roots_y = dty.get_minimal_nodes_covering_dims(dims_y)
#     compl_roots_y = dty.get_minimal_nodes_covering_dims(compl_dims_y)

#     if 1 in [len(roots_x), len(compl_roots_x)] and 1 in [len(roots_y), len(compl_roots_y)]:
#         # Fall a)
#         compl_x = False
#         compl_y = False
#         if len(roots_x) == 1:
#             node_x = roots_x[0]
#         else:
#             node_x = compl_roots_x[0]
#             compl_x = True
#         if len(roots_y) == 1:
#             node_y = roots_y[0]
#         else:
#             node_y = compl_roots_y[0]
#             compl_y = True
#         return _one_node(x, y, dims_x, dims_y, node_x, node_y, compl_x, compl_y)

#     elif len(dims_x) == len(x.shape) or len(dims_y) == len(y.shape):
#         # Fall b)
#         # Vertausche ggf. x und y, sodass immer len(dims_y) == len(y.get_shape()) gilt
#         if len(dims_y) != len(y.shape):
#             x, y = y, x
#             dims_x, dims_y = dims_y, dims_x
#             roots_x, roots_y = roots_y, roots_x
#             compl_roots_x, compl_roots_y = compl_roots_y, compl_roots_x

#         node_x_child = -7
#         if len(roots_x) == 2:
#             # Pruefe dims_x
#             parent0, parent1 = dtx.get_parent(roots_x[0]), dtx.get_parent(roots_x[1])
#             if dtx.get_parent(parent0) == parent1:
#                 node_x_child = parent0
#             elif dtx.get_parent(parent1) == parent0:
#                 node_x_child = parent1
#             elif dtx.get_parent(parent0) == dtx.get_root() and dtx.get_parent(parent1) == dtx.get_root():
#                 node_x_child = parent0

#         if len(compl_roots_x) == 2:
#             # Pruefe compl_dims_x
#             compl_parent0, compl_parent1 = dtx.get_parent(compl_roots_x[0]), dtx.get_parent(compl_roots_x[1])
#             if dtx.get_parent(compl_parent0) == compl_parent1:
#                 node_x_child = compl_parent0
#             elif dtx.get_parent(compl_parent1) == compl_parent0:
#                 node_x_child = compl_parent1
#             elif dtx.get_parent(compl_parent0) == dtx.get_root() and dtx.get_parent(compl_parent1) == dtx.get_root():
#                 node_x_child = compl_parent0

#         # Kontraktion ist nicht moeglich
#         if node_x_child == -7:
#             raise RuntimeError(
#                 "Die Kontraktion kann fuer diesen Fall nicht im hierarchischen Tuckerformat berechnet werden."
#             )

#         return _two_nodes(x, y, dims_x, dims_y, node_x_child)
#     else:
#         raise RuntimeError(
#             "Die Kontraktion kann fuer diesen Fall nicht im hierarchischen Tuckerformat berechnet werden."
#         )


# def _two_nodes(x: HTTensor, y: HTTensor, dims_x: list, dims_y: list, node_x_child: tuple[int, ...]):
#     """Compute the tensordot for the two nodes case.

#     Hinweis: Das ist eine interne Funktion der Funktion HTTensor.tensordot
#     Diese Funktion deckt Fall 1. b) ab. Siehe dazu den Docstring zu HTTensor.tensordot.
#     ______________________________________________________________________
#     Parameter:
#     - x HTucker.HTTensor: Der mit 'y' zu kontrahierende hierarchische Tuckertensor.
#     - y HTucker.HTTensor: Der mit 'x' zu kontrahierende hierarchische Tuckertensor.
#     - dims_x [int,...]: Die zu kontrahierenden Dimensionen von 'x'.
#     - dims_y [int,...]: Die zu kontrahierenden Dimensionen von 'y'.
#     - node_x_child (int,...): Knoten des Dimensionsbaums von 'x'. Einer der beiden Elternknoten aus Fall 1. b).
#     ______________________________________________________________________
#     Output:
#     (HTucker.HTTensor,): Der resultierende kontrahierte hierarchische Tuckertensor.

#     """
#     # Der Knoten node_x_child soll rechtes Kind der Wurzel sein
#     x.change_root(node=node_x_child, lr="right")

#     # Fuer die Lesbarkeit
#     dtx = x.dtree

#     # Berechne, wie sich 'dims_x' auf den linken und rechten subtree der Wurzel von 'x' aufteilen
#     left, right = dtx.get_children(dtx.get_root())
#     left_l, right_l = dtx.get_left(left), dtx.get_left(right)
#     left_lr = 0 if not set(left_l).isdisjoint(set(dims_x)) else 1
#     right_lr = 0 if not set(right_l).isdisjoint(set(dims_x)) else 1
#     dims_x_left = [dim for dim in dims_x if dim in left]
#     dims_x_right = [dim for dim in dims_x if dim in right]

#     # Teile entsprechend auch dims_y auf
#     dims_y_left = [dims_y[i] for i in [idx for idx in range(len(dims_x)) if dims_x[idx] in dims_x_left]]
#     dims_y_right = [dim for dim in dims_y if dim not in dims_y_left]
#     roots_y_left = y.dtree.get_minimal_nodes_covering_dims(dims=dims_y_left)
#     roots_y_right = y.dtree.get_minimal_nodes_covering_dims(dims=dims_y_right)
#     if len(roots_y_left) == 1:
#         y.change_root(node=roots_y_left[0], lr="left")
#     elif len(roots_y_right) == 1:
#         y.change_root(node=roots_y_right[0], lr="right")
#         pass
#     else:
#         raise RuntimeError(
#             "Die Kontraktion kann fuer diesen Fall nicht im hierarchischen Tuckeformat berechnet werden."
#         )

#     # Berechne nun die Kontraktionen der subtrees
#     x_left = deepcopy(x)
#     x_left.change_root(node=dtx.get_children(left)[left_lr], lr="right")
#     M_left = _get_contracted_connection_tensor(x_left, y, dims_x_left, dims_y_left)
#     x_right = deepcopy(x)
#     x_right.change_root(node=dtx.get_children(right)[right_lr], lr="right")
#     M_right = _get_contracted_connection_tensor(x_right, y, dims_x_right, dims_y_right)

#     # Kombiniere M_left und M_right zur Gesamtkontraktion
#     My = M_left @ y.B[y.dtree.get_root()].squeeze(dim=2) @ M_right.T
#     B = torch.tensordot(x.B[right], x.B[x.dtree.get_root()].squeeze(dim=2), dims=[[2], [1]])  # type: ignore
#     B = torch.tensordot(My, B, dims=[[1], [right_lr]])  # type: ignore
#     if right_lr:
#         B = torch.swapaxes(B, 0, 1)
#     M = torch.tensordot(x.B[left], B, dims=[[2, left_lr], [2, right_lr]])  # type: ignore
#     del B

#     # Setze den finalen hierarchischen Tuckertensor zusammen
#     # 1) Konstruiere den resultierenden Dimensionsbaum
#     # 1.1) Knotenhierarchie
#     root = tuple(dim for dim in dtx.get_root() if dim not in dims_x)
#     new_nodes = {root: [dtx.get_children(left)[1 - left_lr], dtx.get_children(right)[1 - right_lr]]}
#     subtree_left = dtx.get_subtree(dtx.get_children(left)[1 - left_lr])
#     subtree_right = dtx.get_subtree(dtx.get_children(right)[1 - right_lr])
#     new_nodes.update(subtree_left.nodes)
#     new_nodes.update(subtree_right.nodes)
#     # 1.2) Knotenmapping berechnen
#     old2new = _get_node_mapping(subtree_left, dims_x)
#     old2new.update(_get_node_mapping(subtree_right, dims_x))
#     new_root = old2new[new_nodes[root][0]] + old2new[new_nodes[root][1]]
#     old2new[root] = new_root
#     # 1.3) Knotenmapping anwenden
#     new_nodes = {old2new[k]: [old2new[vv] for vv in v] for k, v in new_nodes.items()}
#     # 1.4) Baum erzeugen
#     dtree = DimTree(new_nodes)
#     # 2) Blattmatrixdict und Transfertensordict vorbereiten
#     U = {old2new[leaf]: x.U[leaf] for leaf in subtree_left.get_leaves()}
#     U.update({old2new[leaf]: x.U[leaf] for leaf in subtree_right.get_leaves()})
#     B = {old2new[node]: x.B[node] for node in subtree_left.get_inner_nodes()}
#     B.update({old2new[node]: x.B[node] for node in subtree_right.get_inner_nodes()})
#     B[new_root] = M.unsqueeze(dim=2)
#     # 3) Erstellen des resultierenden HTTensor Objekts
#     z = type(x)(U=U, B=B, dtree=dtree, is_orthog=False)
#     return z


# def _one_node(x, y, dims_x, dims_y, node_x, node_y, compl_x, compl_y):
#     """Compute the tensordot for the one node case.

#     Hinweis: Das ist eine interne Funktion der Funktion HTTensor.tensordot
#     Diese Funktion deckt Fall 1. a) ab. Siehe dazu den Docstring zu HTTensor.tensordot.
#     ______________________________________________________________________
#     Parameter:
#     - x HTucker.HTTensor: Der mit 'y' zu kontrahierende hierarchische Tuckertensor.
#     - y HTucker.HTTensor: Der mit 'x' zu kontrahierende hierarchische Tuckertensor.
#     - dims_x [int,...]: Die zu kontrahierenden Dimensionen von 'x'.
#     - dims_y [int,...]: Die zu kontrahierenden Dimensionen von 'y'.
#     - node_x (int,...): Knoten des Dimensionsbaums von 'x', dessen Subtree entweder genau die zu kontrahierenden
#                         Dimensionen oder genau die nicht zu kontrahierenden Dimensionen enthaelt.
#     - node_y (int,...): Knoten des Dimensionsbaums von 'y', dessen Subtree entweder genau die zu kontrahierenden
#                         Dimensionen oder genau die nicht zu kontrahierenden Dimensionen enthaelt.
#     - compl_x bool: Zeigt an, ob der Subtree von 'node_x' alle Dimensionen aus 'dims_x' oder deren Komplement enthaelt.
#     - compl_y bool: Zeigt an, ob der Subtree von 'node_y' alle Dimensionen aus 'dims_y' oder deren Komplement enthaelt.
#     ______________________________________________________________________
#     Output:
#     (HTucker.HTTensor,): Der resultierende kontrahierte hierarchische Tuckertensor.

#     """
#     # Merke die Wurzeln der Dimensionsbaeume von x und y
#     rootx, rooty = x.dtree.get_root(), y.dtree.get_root()

#     # Das rechte Kind der Wurzel soll alle zu kontrahierenden Dimensionen enthalten
#     # Dies gilt fuer x und y
#     x._change_root(node=node_x, lr="left") if compl_x else x._change_root(node=node_x, lr="right")
#     y._change_root(node=node_y, lr="left") if compl_y else y._change_root(node=node_y, lr="right")

#     # Falls _change_root eine neue Wurzel erzeugt hat, muessen die zu kontrahierenden Dimensionen in dims_x/y
#     # entsprechend angepasst werden
#     squeeze_left = False
#     if node_x == rootx:
#         if compl_x:
#             # Die in _change_root hinzugefuegte Singleton Dimension wird fuer das anschliessende Kontrahieren markiert
#             dims_x = [len(x.get_shape()) - 1]
#         else:
#             dims_x = [dim + 1 for dim in dims_x]
#             squeeze_left = True
#     squeeze_right = False
#     if node_y == rooty:
#         if compl_y:
#             # Die in _change_root hinzugefuegte Singleton Dimension wird fuer das anschliessende Kontrahieren markiert
#             dims_y = [len(x.get_shape()) - 1]
#         else:
#             dims_y = [dim + 1 for dim in dims_y]
#             squeeze_right = True

#     # Kontrahiere nun den linken/rechten Subtree von x mit dem linken/rechten Subtree von y
#     C = _get_contracted_connection_tensor(x, y, dims_x, dims_y)

#     # Konstruiere den resultierenden Dimensionsbaum
#     # 1) Erhalte linke subtrees von x und y
#     dtx = x.dtree.get_subtree(x.dtree.get_left(x.dtree.get_root()))
#     dty = y.dtree.get_subtree(y.dtree.get_left(y.dtree.get_root()))
#     # 2) Kontrahiere die Wurzeln von x und y mit C
#     new_root_tensor = torch.tensordot(x.B[x.dtree.get_root()], C, dims=[[1], [0]])  # type: ignore
#     new_root_tensor = torch.tensordot(new_root_tensor, y.B[y.dtree.get_root()], dims=[[2], [1]])  # type: ignore
#     new_root_tensor = torch.squeeze(new_root_tensor, dim=[1, 3])
#     new_root_tensor = torch.unsqueeze(new_root_tensor, dim=2)
#     # 3) Berechne Knotenmapping
#     old2new_x = _get_node_mapping(dtx, dims_x)
#     old2new_y = _get_node_mapping(dty, dims_y)
#     offset_y = len(dtx.get_leaves())
#     old2new_y = {k: tuple(dim + offset_y for dim in v) for k, v in old2new_y.items()}
#     # 4) Erstelle neue Knotenhierarchie
#     # 4.1) Fuege Hierarchie des uebrig gebliebenen subtrees von x hinzu
#     new_nodes = {old2new_x[k]: [old2new_x[dim] for dim in v] for k, v in dtx.nodes.items()}
#     # 4.2) Fuege Hierarchie des uebrig gebliebenen subtrees von y hinzu
#     new_nodes.update({old2new_y[k]: [old2new_y[dim] for dim in v] for k, v in dty.nodes.items()})
#     # 4.3) Erzeuge verbindende Wurzel und fuege sie ein
#     new_root = old2new_x[dtx.get_root()] + old2new_y[dty.get_root()]
#     new_nodes[new_root] = [old2new_x[dtx.get_root()], old2new_y[dty.get_root()]]
#     # 4.4) Erzeuge Dimensionsbaum
#     dtree = DimTree(nodes=new_nodes)

#     # Konstruiere Blattmatrix dict
#     U = {old2new_x[leaf]: x.U[leaf] for leaf in dtx.get_leaves()}
#     U.update({old2new_y[leaf]: y.U[leaf] for leaf in dty.get_leaves()})

#     # Konstruiere Transfertensor dict
#     B = {old2new_x[node]: x.B[node] for node in dtx.get_inner_nodes()}
#     B.update({old2new_y[node]: y.B[node] for node in dty.get_inner_nodes()})
#     B[dtree.get_root()] = new_root_tensor

#     # Erzeuge den zugehoerigen hierarchischen Tuckertensor
#     z = type(x)(U=U, B=B, dtree=dtree, is_orthog=False)

#     # Entferne evtl. vorhandene Singleton Dimensions
#     if squeeze_left and squeeze_right:
#         z.squeeze()
#     elif squeeze_left:
#         z.squeeze(dims=0)
#     elif squeeze_right:
#         z.squeeze(dims=len(z.get_shape()) - 1)

#     return z


# def _get_contracted_connection_tensor(x, y, dims_x, dims_y):
#     """Compute contracted tensor.

#     Hinweis: Dies ist eine intere Funktion der Funktion HTTensor.tensordot.
#     Kontrahiert den rechten Subtree von 'x' mit (Teilen von) 'y'. Dabei geben 'dims_x' und 'dims_y' an, welche
#     # Dimensionen aus 'x' und 'y' kontrahiert werden.
#     """
#     # Die zu kontrahierenden Subtrees von x und y
#     dtx = x.dtree.get_subtree(x.dtree.get_right(x.dtree.get_root()))
#     dty = y.dtree

#     # Initialisiere Mapping der Knoten von dtx und dty
#     # Fuege dem Mapping zunaechst die Blaetter hinzu
#     x2y = {(dims_x[i],): (dims_y[i],) for i in range(len(dims_x))}

#     # Enthaelt die Kontraktionen
#     C = {}

#     # Traversiere bottom up
#     for level in range(dtx.get_depth(), -1, -1):
#         for nodex in dtx.get_nodes_of_lvl(level):
#             if dtx.is_leaf(nodex):
#                 nodey = x2y[nodex]
#                 C[nodex] = x.U[nodex].T @ y.U[nodey]

#             else:
#                 lx, rx = dtx.get_children(nodex)
#                 ly, ry = x2y[lx], x2y[rx]

#                 if dty.get_parent(ly) != dty.get_parent(ry):
#                     raise RuntimeError("Die Dimensionsbaeume von 'x' und 'y' sind nicht kompatibel.")

#                 nodey = dty.get_parent(ly)

#                 # Aktualisiere Knoten Mapping
#                 x2y[nodex] = nodey

#                 if dty.is_left(ly):
#                     C_buf = torch.tensordot(C[lx], y.B[nodey], dims=[[1], [0]])  # type: ignore
#                 else:
#                     C_buf = torch.tensordot(C[lx], y.B[nodey], dims=[[1], [1]])  # type: ignore
#                 C_buf = torch.tensordot(C[rx], C_buf, dims=[[1], [1]])  # type: ignore
#                 C[nodex] = torch.tensordot(x.B[nodex], C_buf, dims=[[0, 1], [1, 0]])  # type: ignore

#                 del C[lx]
#                 del C[rx]
#     return C[dtx.get_root()]


# def _get_node_mapping(dtree, dims):
#     """Return node mapping.

#     Hinweis: Dies ist eine interne Funktion der Funktion HTTensor.tensordot.
#     """
#     # Baue das Mapping beginnend mit den Blaettern auf
#     old2new = {}
#     for leaf in dtree.get_leaves():
#         leaf_dim = leaf[0]
#         offset = len([dim for dim in dims if dim < leaf_dim])
#         old2new[leaf] = (leaf_dim - offset,)
#     # Darauf aufbauend kann nun das Mapping der inneren Knoten berechnet werden
#     for level in range(dtree.get_depth() - 1, -1, -1):
#         for node in dtree.get_nodes_of_lvl(level):
#             if dtree.is_leaf(node):
#                 continue
#             l, r = dtree.get_children(node)
#             old2new[node] = old2new[l] + old2new[r]
#     return old2new
