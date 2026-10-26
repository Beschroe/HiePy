# The HT format

A tensor $x \in \mathbb{R}^{n_0 \times \dots \times n_{d-1}}$ has $n_0 \cdots n_{d-1}$ entries, which grows
exponentially with the order $d$. The Hierarchical Tucker (HT) format stores it instead as a collection of small
matrices and 3-way tensors, arranged along a binary tree over the modes. For tensors with low *hierarchical ranks*,
the storage grows only linearly with $d$.

## Dimension tree

The modes $0, \dots, d-1$ are split recursively into two groups until every group holds a single mode. hiepy calls
this the dimension tree (`DimTree`); its nodes are tuples of mode indices. Factories and `to_htt` use a balanced
tree:

```python
>>> import torch
>>> import hiepy
>>> torch.set_default_dtype(torch.float64)
>>> x = hiepy.randn((4, 5, 6, 7), rank=3, generator=torch.Generator().manual_seed(0))
>>> print(x.dtree)  # doctest: +NORMALIZE_WHITESPACE
    (0, 1, 2, 3)
  (0, 1)    (2, 3)
 (0,) (1,) (2,) (3,)
>>> x.dtree.get_children((0, 1, 2, 3))
((0, 1), (2, 3))

```

## Components

Each **leaf** $t = (i,)$ holds a matrix $U_t \in \mathbb{R}^{n_i \times r_t}$, each **inner node** $t$ with children
$t_l, t_r$ a transfer tensor $B_t \in \mathbb{R}^{r_{t_l} \times r_{t_r} \times r_t}$. The root has $r = 1$.

```python
>>> {t: tuple(u.shape) for t, u in x.U.items()}
{(0,): (4, 3), (1,): (5, 3), (2,): (6, 3), (3,): (7, 3)}
>>> {t: tuple(b.shape) for t, b in x.B.items()}
{(2, 3): (3, 3, 3), (0, 1): (3, 3, 3), (0, 1, 2, 3): (3, 3, 1)}

```

The components define a *frame* $U_t$ for every node, recursively from the leaves:

$$
(U_t)_{(i_l, i_r),\,k} = \sum_{a, b} (U_{t_l})_{i_l, a}\, (U_{t_r})_{i_r, b}\, (B_t)_{a, b, k},
$$

where $i_l$ and $i_r$ run over the entries of the modes of $t_l$ and $t_r$. The frame of the root is the tensor
itself. For the tree above, written out:

$$
x_{i_0 i_1 i_2 i_3} = \sum_{a,b,c,e,p,q} (U_0)_{i_0 a} (U_1)_{i_1 b} (B_{01})_{a b p}
\, (U_2)_{i_2 c} (U_3)_{i_3 e} (B_{23})_{c e q} \, (B_{0123})_{p q 0}.
$$

```python
>>> U, B = x.U, x.B
>>> E = torch.einsum(
...     "ia,jb,abp,kc,le,ceq,pqr->ijkl",
...     U[(0,)], U[(1,)], B[(0, 1)], U[(2,)], U[(3,)], B[(2, 3)], B[(0, 1, 2, 3)],
... )
>>> bool(torch.allclose(E, x.full()))
True

```

## Ranks and storage

`x.rank` reads the ranks $r_t$ off the component shapes:

```python
>>> x.rank
{(0, 1, 2, 3): 1, (0,): 3, (1,): 3, (2,): 3, (3,): 3, (0, 1): 3, (2, 3): 3}

```

With mode sizes at most $n$ and ranks at most $r$, the components need $\mathcal{O}(d n r + d r^3)$ numbers instead
of $n^d$. Every function documents in its *Notes* how it changes the ranks and what it costs.

These are the ranks of the *representation*. They can be larger than necessary, for example after a sum:
`orthog` removes redundancy that the component sizes allow, and `truncate` reduces the ranks to the accuracy you
request (see [Truncation](truncation.md)).

```python
>>> max((x + x).rank.values())
6
>>> max((x + x).truncate({"rtol": 1e-12})[0].rank.values())
3

```

## Other trees

Any binary tree over the modes is allowed, for example an unbalanced one:

```python
>>> tree = hiepy.DimTree(
...     (0, 1, 2, 3),
...     {(0, 1, 2, 3): ((0,), (1, 2, 3)), (1, 2, 3): ((1,), (2, 3)), (2, 3): ((2,), (3,))},
... )
>>> print(tree)  # doctest: +NORMALIZE_WHITESPACE
              (0, 1, 2, 3)
        (0,)             (1, 2, 3)
                       (1,)     (2, 3)
                               (2,) (3,)

```

An `HTTensor` with such a tree is built from its components, `hiepy.HTTensor(U, B, tree)`. Removing modes with
integer indexing or `squeeze` also changes the tree. Operations on two tensors (`+`, `-`, `*`, `@`) require
identical trees and shapes.
