from collections import defaultdict, Counter
from itertools import combinations, permutations
from typing import Dict, Iterable, Optional, Tuple, Any
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


# Keep the same mapping_key format the original used (pairs). Also provide an image-only key for compatibility.
def mapping_key_pairs(mapping: Dict[Any, Any], domain: Optional[Iterable[Any]] = None) -> Tuple[Tuple[Any, Any], ...]:
    """
    Canonical key: tuple of (element, image) pairs in sorted(domain) order.
    This matches the original mapping_key that returned ((x, mapping[x]), ...).
    """
    if domain is None:
        domain = sorted(mapping.keys())
    else:
        domain = sorted(domain)
    return tuple((x, mapping.get(x, x)) for x in domain)


def mapping_key_images(mapping: Dict[Any, Any], domain: Optional[Iterable[Any]] = None) -> Tuple[Any, ...]:
    """
    Compatibility key: tuple of images only, in sorted(domain) order.
    Some earlier variants used this representation; we check both forms.
    """
    if domain is None:
        domain = sorted(mapping.keys())
    else:
        domain = sorted(domain)
    return tuple(mapping.get(x, x) for x in domain)


def find_swappable_pairs(colored_lists):
    AllLists = {}
    ListsByElem = {}
    colors = colored_lists.keys()
    for c in colors:
        AllLists[c] = set()
        ListsByElem[c] = defaultdict(set)
        for L in colored_lists[c]:
            fs = frozenset(L)
            AllLists[c].add(fs)
            for i in fs:
                ListsByElem[c][i].add(fs)

    signature = {}
    all_elements = set()
    for c in colors:
        all_elements |= set(ListsByElem[c].keys())
    for i in all_elements:
        signature[i] = tuple(len(ListsByElem[c].get(i, ())) for c in colors)

    sig_groups = defaultdict(list)
    for i, sig in signature.items():
        sig_groups[sig].append(i)

    swappable = []
    for element in sig_groups.values():
        for x, y in combinations(element, 2):
            valid = True
            for c in colors:
                Lx = ListsByElem[c].get(x, set())
                Ly = ListsByElem[c].get(y, set())
                both = Lx & Ly
                A = Lx - both
                B = Ly - both
                if len(A) != len(B):
                    valid = False
                    break
                for L in A:
                    swapped = frozenset((L - {x}) | {y})
                    if swapped not in AllLists[c]:
                        valid = False
                        break
                if not valid:
                    break
                for L in B:
                    swapped = frozenset((L - {y}) | {x})
                    if swapped not in AllLists[c]:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                swappable.append((x, y))
    return swappable


def plot_group_graph(group_lists, group_name="group"):
    """
    group_lists: list of lists (e.g. from colored_lists['red'])
    Displays a bipartite graph of elements vs list-nodes
    """
    G = nx.Graph()
    element_nodes = set()
    list_nodes = []

    for idx, L in enumerate(group_lists):
        list_id = f"L{idx}"
        list_nodes.append(list_id)
        G.add_node(list_id, bipartite=0, label="list")
        for elem in L:
            G.add_node(elem, bipartite=1, label="element")
            G.add_edge(list_id, elem)
            element_nodes.add(elem)

    # Layout: bipartite
    pos = {}
    pos.update((n, (1, i)) for i, n in enumerate(sorted(element_nodes)))   # top row
    pos.update((n, (0, i)) for i, n in enumerate(list_nodes))              # bottom row

    plt.figure(figsize=(8, 5))
    nx.draw(
        G, pos,
        with_labels=True,
        node_size=800,
        node_color=["lightblue" if n in element_nodes else "lightgreen" for n in G.nodes],
        edge_color="gray"
    )
    plt.title(f"Incidence Graph for Group '{group_name}'")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


# # Below are functions for finding permutations with k-cycles up to a certain size.


def multiset_by_label(lists, labels):
    """
    Build per-label multisets of lists where each list is treated as a set (frozenset).
    Returns: dict[label] -> Counter({frozenset(list): count})
    """
    by_label = defaultdict(Counter)
    for lst, lab in zip(lists, labels):
        by_label[lab][frozenset(lst)] += 1
    return by_label


def apply_mapping_to_lists(lists, mapping):
    """Apply element mapping to every list."""
    return [[mapping.get(x, x) for x in lst] for lst in lists]


def preserves_per_label_multisets(lists, labels, mapping):
    """True iff applying mapping preserves each label group's multiset of lists."""
    orig = multiset_by_label(lists, labels)
    permuted = multiset_by_label(apply_mapping_to_lists(lists, mapping), labels)
    return orig == permuted


def mapping_key(mapping, domain=None):
    """
    Canonical tuple key for a mapping so we can store/compare permutations robustly.
    If domain is None, use all keys in 'mapping'.
    """
    if domain is None:
        domain = sorted(mapping.keys())
    return tuple((x, mapping.get(x, x)) for x in sorted(domain))


def find_one_permutation(list_of_dependencies: list[list[int]], coefficients: list[complex] | np.ndarray,
                         previously_found: set | None = None,
                         max_cycle_size: int = 3):
    """
    Brute force search for a single permutation.

    Try to find one nontrivial permutation (as a mapping dict) that preserves all
    label-group multisets. Tries 2-cycles, then 3-cycles, ... up to max_cycle_size.

    Returns a single valid permutation that is not in previously_found.

    Example:
    list_of_dependencies = [
        [1, 2],   # label i
        [1, 4],   # label i
        [5, 3]    # label 1
    ]
    coefficients = [0+1j, 0+1j, 1+0j]
    found = find_one_permutation(list_of_dependencies, coefficients, previously_found=set(), max_cycle_size=4)

    Returns:
    {1: 1, 2: 4, 3: 3, 4: 2, 5: 5}

    meaning that the resulting permutation is a swap of elements 2 and 4.
    """
    if previously_found is None:
        previously_found = set()

    elements = sorted({x for lst in list_of_dependencies for x in lst})
    # Try small cycles first: 2-cycles (swaps), then 3-cycles, etc.
    for k in range(2, max_cycle_size + 1):
        for subset in combinations(elements, k):
            # Generate directed cycles over this subset
            for perm in permutations(subset, k):
                # Require a single k-cycle: perm is a rotation of subset
                # (Otherwise you'd generate all permutations; this prunes a lot.)
                # We'll accept any cycle that moves everything in 'subset'.
                if all(perm[i] != subset[i] for i in range(k)):  # non-identity on subset
                    cyc_map = {subset[i]: perm[i] for i in range(k)}
                    # Leave everything else fixed
                    mapping = {x: cyc_map.get(x, x) for x in elements}
                    key = mapping_key(mapping, domain=elements)
                    if key in previously_found:
                        continue
                    if preserves_per_label_multisets(list_of_dependencies, coefficients, mapping):
                        return mapping
    return None


def brute_force_all_permutations(list_of_dependencies: list[list[int]],
                                 coefficients: list[complex] | np.ndarray):
    """
    Brute force all permutations of elements and return all valid mappings.

    Returns: list of dicts, each dict is a valid automorphism mapping.
    """
    elements = sorted({x for lst in list_of_dependencies for x in lst})
    valid_mappings = []

    for perm in permutations(elements):
        mapping = {elements[i]: perm[i] for i in range(len(elements))}
        if preserves_per_label_multisets(list_of_dependencies, coefficients, mapping):
            valid_mappings.append(mapping)

    return valid_mappings


def permutation_to_swaps(perm_dict):
    """
    Convert a permutation dictionary {old_index: new_index} into a list
    of swaps (tuples of length two).

    The swaps are chosen so that applying them in order will produce
    the permutation.
    """
    swaps = []
    visited = set()

    for start in perm_dict:
        if start in visited or perm_dict[start] == start:
            continue

        # Follow the cycle starting at 'start'
        cycle = []
        current = start
        while current not in visited:
            visited.add(current)
            cycle.append(current)
            current = perm_dict[current]

        # Break cycle into swaps (start with first element fixed)
        for i in range(1, len(cycle)):
            swaps.append((cycle[0], cycle[i]))

    return swaps


def find_swapped_dependent_elements(swaps, elements):
    """
    Given a list of swaps (elementary basis swaps) and a list of elements,
    return which dependent elements (list-of-lists) are swapped.
    """
    # Defensive check
    for s in swaps:
        if not (isinstance(s, tuple) and len(s) == 2):
            raise ValueError(f"Each swap must be a 2-tuple, got {s!r}")

    # Build mapping from swaps
    mapping = {a: b for a, b in swaps}
    mapping.update({b: a for a, b in swaps})

    def apply_map(lst):
        return sorted(mapping.get(x, x) for x in lst)

    mapped = [apply_map(lst) for lst in elements]

    swapped_pairs = []
    used = set()
    for i, m in enumerate(mapped):
        if i in used:
            continue
        for j, orig in enumerate(elements):
            if sorted(orig) == m:
                if i != j:
                    swapped_pairs.append((i, j))
                    used.add(j)
                break
    return swapped_pairs


# if __name__ == "__main__":
#     # --- Example usage ---
#     data2 = {
#         'red': [[1, 2], [1, 4], [2, 6]],
#         'blue': [[5, 3]],
#         'green': [[6, 7]]
#     }

#     print("Swappable pairs:", find_swappable_pairs(data2))
#     plot_group_graph(data2['red'], group_name="red")

#     lists = [
#         [1, 2],   # label i
#         [1, 4],   # label i
#         [5, 3]    # label 1
#     ]
#     labels = [0 + 1j, 0 + 1j, 1 + 0j]

#     found = find_one_permutation(lists, labels, previously_found=set(), max_cycle_size=4)
#     print("Found permutation:", found)
#     if found:
#         print("Preserves groups:", preserves_per_label_multisets(lists, labels, found))
