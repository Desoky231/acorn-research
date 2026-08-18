# SIEVE: Effective Filtered Vector Search with Collection of Indexes

Zhaoheng Li

Note:
Work done during internship at Bytedance.

Affiliation:
UIUC

email:
zl20@illinois.edu

,

Silu Huang

Affiliation:
Bytedance Inc.

email:
silu.huang@bytedance.com

,

Wei Ding

Affiliation:
Bytedance Inc.

email:
wei.ding@bytedance.com

,

Yongjoo Park

Affiliation:
UIUC

email:
yongjoo@illinois.edu

and

Jianjun Chen

Affiliation:
Bytedance Inc.

email:
jianjun.chen@bytedance.com

###### Abstract.

Real-world tasks such as
recommending videos tagged
kids
can be reduced to finding
similar
vectors associated with
hard
predicates.
This task,
filtered vector search
, is
challenging
as prior state-of-the-art graph-based (unfiltered) similarity search techniques degenerate when hard constraints are considered: effective graph-based filtered similarity search relies on sufficient connectivity for reaching similar items
within a few hops.
To consider predicates,
recent works propose modifying graph traversal
to visit only items that satisfy predicates.
However, they fail to offer the just-a-few-hops property
for a wide range of predicates:
they must restrict predicates significantly
or lose efficiency
if only few items satisfy predicates.

We propose an opposite approach:
instead of constraining traversal,
we build many indexes each serving different predicate forms.
For effective construction,
we devise a three-dimensional analytical model
capturing relationships among index size, search time, and recall, with which we follow a workload-aware approach to pack as many useful indexes as possible into a collection.
At query time,
the analytical model is employed yet again
to discern the one that offers the fastest search
at a given recall.
We show superior performance and support
on datasets with varying selectivities and forms: our approach achieves up to 8.06
$\times$
speedup while having as low as 1% build time versus other indexes, with less than 2.15
$\times$
memory of a standard HNSW graph and modest knowledge of past workloads.

PVLDB Reference Format:

Zhaoheng Li, Silu Huang, Wei Ding, Yongjoo Park, Jianjun Chen.  PVLDB, 18(11): 4723 - 4736, 2025.

doi:10.14778/3749646.3749725

†
†

This work is licensed under the Creative Commons BY-NC-ND 4.0 International License. Visit
https://creativecommons.org/licenses/by-nc-nd/4.0/
to view a copy of this license. For any use beyond those covered by this license, obtain permission by emailing
info@vldb.org
. Copyright is held by the owner/author(s). Publication rights licensed to the VLDB Endowment.

Proceedings of the VLDB Endowment, Vol. 18, No. 11 ISSN 2150-8097.

doi:10.14778/3749646.3749725

PVLDB Artifact Availability:

The source code, data, and/or other artifacts have been made available at
https://github.com/BillyZhaohengLi/SIEVE-vldb25
.

## 1.Introduction

Finding
semantically similar
items
satisfying
hard constraints
is a common task.
Moms may search for videos (semantic)
tagged “safe-for-kids” (hard).
Online shoppers may search for costumes (semantic)
with a specific price range (hard)
(
Zuo et al. 2024
)
.
This task is called

filtered vector search
:
we query similar
vectors
—encoding semantics—associated with
hard
predicates.
The problem has been increasingly studied
(
Patel et al. 2024
;
Gupta et al. 2023
;
Douze et al. 2024
;
Gollapudi et al. 2023
;
Wu et al. 2022
;
Mohoney et al. 2023
;
Wang et al. 2022
;
Landrum et al. 2025
;
Simhadri et al. 2024
;
Zuo et al. 2024
;
Engels et al. [n.d.]
;
Sanca and Ailamaki 2024
)
as quality vector embeddings become available via modern ML models
(
Meta [n.d.]
;
Microsoft [n.d.]
;
OpenAI [n.d.]
)
.

Some works tackle filtered vector search by
constraining
graph-based approximate nearest neighbor (ANN) indexes for
unfiltered vector search

(
Malkov and Yashunin 2018
;
Jayaram Subramanya et al. 2019
;
Fu et al. 2017
)
:
FilteredVamana
interleaves graph traversal and filter evaluations;
ACORN
(
Patel et al. 2024
)

induces query-time subgraphs by visiting only predicate-passing nodes.
These works outperform
naïve methods like
pre-filtering
,
which uses a (slow) linear scan for similarity computations.
Graph-based methods
navigate items via edges to reach targets with a few hops. Effective filtered vector search aims to serve queries with
compact
graphs; if the property—
small world
—is lost, graph traversal will lose efficiency.

Unfortunately,

existing graph-based methods fail to offer the small-world property for low-selectivity predicates,

thus delivering poorer performance.
Moreover,
we cannot simply use pre-filtering
as a linear scan is still too costly
unless the selectivity is
too low
.
This selectivity band is called the “unhappy middle”
(
Gupta et al. 2023
)
.

FilteredVamana
aims to mitigate this by linking attribute-sharing vectors into local, dense, per-filter subgraphs, but requires restricting filter forms
(
Gollapudi et al. 2023
)
.
ACORN
(
Patel et al. 2024
)
supports general predicates at a cost: its induced subgraph can provably lose the small-world property if it becomes too sparse
(
Amaral et al. 2000
;
Kashyap and Ambika 2019
;
Malkov and Yashunin 2018
)
.
We conjecture that a single graph
is insufficient for handling
all
predicates,
whose support may overlap with one another in a complex way.
We may need multiple graphs,
each specialized for different predicate sets.

##### Our Goal

We aim to offer
compact
graphs
for nearly all filtered queries with varying selectivities or forms by building an
index collection
.
A collection is more expressive than one index.
By leveraging
filter stability
in real-world filtered vector search workloads
(
Mohoney et al. 2023
;
Sun et al. 2014
)
, we can tailor indexes to observed workloads to maximize expected search quality.
Each index can serve multiple predicates: a graph, e.g., built for
stars=1--3
, can also serve
stars=1
if it is
dense enough
for the sub-predicate.
An index collection requires more memory than one index;
yet, indexes are relatively small versus raw data, i.e., high-dimensional vectors.
In our experiments,
hundreds of (small) additional indexes
took only as much memory as one dataset-wide index,
while they boost performance significantly
versus relying on one index.
Our proposed index collection can succeed
if it offers high-quality filtered search to nearly all queries,
each with a compact graph,
while being memory-efficient.

##### Challenge

Building an effective index collection is challenging
due to conflicting goals.
For construction, graphs can trade recall for smaller size (
), allowing more indexes and coverage.
Yet, graphs must be dense enough for high recall.
Likewise, for querying, we can trade search speed for higher recall (
).
This relationship must be quantified
to find which index to use for a given query.
These unique properties distinguishes our task from existing problems
(
Sharma et al. 2016
;
Song et al. 2019
)
.
For example,
materialized view selection targets
exact
querying,
whereas ANN is
approximate
with speed/recall trade-offs.

##### Our Approach

Our framework, called
SIEVE
(
S
et of
I
ndexes for
E
fficient
V
ector
E
xploration), builds an index collection tailored to an observed query workload
to maximize expected throughput
with a memory budget and specified recall.
Every candidate index is assigned
benefit
—its marginal performance gain when added to a collection, and memory
cost
, with which we build a collection.
This approach isn’t new;
what’s new are
(1) how to estimate the benefit/cost
and (2) how to serve queries with the index collection.

First, we design an analytical, predicate form-agnostic benefit/cost model capturing three-dimensional relationships among
index size, search time, and recall,
allowing us to find the minimal (i.e., most sparse) graph satisfying a specific recall.
Since each index is smaller,
more indexes are allowed within a memory budget,
thus accelerating search for more predicates.
Our model is based on empirical observations
and existing small-world network theories
(
Amaral et al. 2000
)
.

Second, query serving dynamically chooses the fastest-searching index given a specific recall.
This query-time selection is needed
since our indexes may overlap: a query may be servable by multiple indexes.
For optimal selection,
we again employ our model
to determine (1) which index to use,
and (2) its search parameterization.

##### Difference from Existing Work

SIEVE
significantly differs from existing vector search works (
). Versus works in MV selection
(
Zhang et al. 2001
;
Agrawal et al. 2000
)
, partitioning
(
Sun et al. 2014
;
Yang et al. 2020
)
and query rewriting
(
Goldstein and Larson 2001
;
Godfrey et al. 2009
)
for exact queries,
SIEVE
’s optimizations notably considers recall, and performs theory-driven index tuning (
section

4.2
) and dynamic, recall-aware serving (
section

5.2
) for desired memory/speed/recall tradeoffs.

##### Contributions

We propose
SIEVE
, an indexing framework for filtered vector search (
section

3
) with the following contributions:

•

Index Selection:
We introduce a three-dimensional cost model for evaluating the search speed, recall, and memory cost of vector search strategies, which we use to jointly perform index selection and parameterization under bounded memory. (
section

4
)

•

Query Serving:
We utilize our derived cost model to derive a dynamic search strategy that selects the most efficient serving method and parameterization at any target recall. (
section

5
)

•

Effective Filtered Vector Search
: We show via experimentation that
SIEVE
achieves up to 8.06
$\times$
speedup over existing indexes with
$<$
2.15
$\times$
memory of a standard HNSW and modest past workload knowledge on diverse query filter formats. (
section

7
)

## 2.Motivation

SIEVE
builds on HNSW
(
Malkov and Yashunin 2018
)
, a performant unfiltered vector index
(
cmuparlay 2024
;
Zilliz 2024
)
. We describe HNSW (
section

2.1
), how works have (ineffectively) extended it to filtered search (
section

2.2
), and our ideas for building and using an HNSW index collection for effective filtered search (
section

2.3
).

### 2.1.HNSW Graph

HNSW is a graph-based vector index
(
Malkov and Yashunin 2018
)
which combines the idea of
small-world graphs

(
Amaral et al. 2000
)
and skip-lists
(
Wikipedia 2024
)
to create a multi-layer graph structure for effective similarity search on vector datasets.

##### HNSW Graph Structure

An HNSW graph consists of multiple layered small-world graphs
(
Amaral et al. 2000
)
. The topmost (entry) layer contains the fewest vectors and features long edge lengths, facilitating long-range vector space travel; the bottom (base) layer contains all vetors and features short edge lengths, representing local neighborhoods.
HNSW graphs are built by incrementally inserting vectors: each vector is linked to a number of neighbors in each layer, controlled by a construction parameter
$M$
, which acts as an outdegree limit and ensures vectors connect to other similar vectors in each layer.

##### HNSW Search

Given a query vector, graph layers are traversed from top to bottom, using long high-layer links to go to neighborhoods with similar data vectors, then using short low-layer links to find top-k choices.

presents the per-layer search algorithm.

### 2.2.Existing Filtering Methods Underperform

While the original HNSW graph proposal did not consider performing filtered vector search, a number of HNSW-based filtered search methods have been proposed, which this section will overview.

##### Post-Search Filtering

For filtered top-
$k$
queries with selectivity
$sel$
, the graph can be over-searched for top-
$k/sel$
vectors. Non-matching results are dropped expecting that
$k$
of top-
$k/sel$
results remain: if not, another top-
$2k/sel$
search is performed, and so on.

##### Result-Set-Filtering

hnswlib

(
Naidan et al. 2015
)
evaluates the query filter during HNSW search before adding candidates into the top-k result set (line 13,
). While this improves over post-search filtering by returning
$k$
satisfying results in one pass, each candidate still has a
$(1\!-\!sel)$
chance to be rejected from the top-k set by the filter. Hence, while recall is negligibly impacted, search time scales inversely with selectivity
(
), effectively still underperforming when
$sel$
is low but with many points for pre-filter search (e.g., large datasets
(
big-ann benchmarks 2024
)
).

1
1

1

In particular,
hnswlib
, ACORN (and our work) all implement filtering via a commonly-used bitmap-based filtering method that in principle handles arbitrary predicates: it assigns IDs to inserted vectors, then computes a binary ID
$->$
{0,1} mapping from IDs of vectors that pass the filter w.r.t. scalar attributes (e.g., via an external RDBMS,
section

6
).

##### Other Filter Application Methods

ACORN
(
Patel et al. 2024
)
applies filtering at neighbor expansion (line 6,
), effectively searching in an induced subgraph of satisfying vectors in the HNSW graph. However, as subgraph induction is equivalent to edge and node removal, the subgraph can lose small-world properties, notably connectivity
(
Xiao et al. 2024
)
, required for effective search if it is too sparse
(
Newman et al. 2000
)
; searching as is with
can result in early stops and low recall. Hence, ACORN modifies both HNSW construction and search, notably expanding into 2-hop neighbors to avoid subgraph sparsity. However, ACORN can still underperform when even the 2-hop subgraph is sparse.
As we will show via experimentation (
section

7.2
), result-set filtering sometimes outperforms ACORN and vice versa.
SIEVE
uses result-set-filtering in its HNSW indexes as it is applicable without specialized graph construction, which may incur excessive time-to-index (TTI,
section

7.3
) and limit discussion to result-set-filtering in the following sections. However,
SIEVE
can also use ACORN’s filtering instead given minor adjustments.

### 2.3.SIEVE’s Intuition for Faster Search

Existing HNSW-based filtering methods underperform on “unhappy-middle“ selectivities.
SIEVE
aims to mitigate this by workload-driven fitting of a HNSW (sub)index collection over data subsets in which these queries can be effectively served from their matching points being dense in the subindexes. As mentioned in
section

1
, building and using HNSW graphs involves speed/recall/memory trade-offs (
); hence,
SIEVE
should decide both
which
and
how
to build and use the index collection; this section describes our intuitions.

##### Three-Dimensional Modeling

Without loss of generality,
SIEVE
treats recall and memory as constraints and optimizes for speed, as users often have ① bounded memory for indexing
(
Li et al. 2023
;
Sun et al. 2016
;
Sun et al. 2014
)
and ② target recalls (e.g., SLOs
(
Sharma et al. 2016
;
Song et al. 2019
)
).
This differs from MV Selection for (exact) querying which is typically only memory-constrained;
SIEVE
’s intuition is that with theory-driven modeling, the recall dimension can be
reduced
by reasoning
how
an index should be built (explained shortly) for different target recalls. Then,
SIEVE
can use established methods to choose
which
indexes to build with bounded memory to maximize serving speed. Finally, for serving,
SIEVE
can determine with similar modeling
which
and
how
to use built indexes for fastest search under a possibly new target recall.

##### How to Build Indexes?

Each subindex’s memory size scales linearly with the (1) indexed vector count and (2) density-controlling construction parameter
$M$
(
section

2.1
).
$M$
can be tuned for different memory/recall tradeoffs: higher
$M$
increases both memory size and recall (from increased density) and vice versa (
). A target recall effectively dictates the
lowest
$M$
value
each index can be built with;
2
2

2

SIEVE
optimizes for average recall as to the best of our knowledge, there exists no method that guarantees
absolute
, per-query recall, as query hardness can vary
(
Wang et al. 2024
)
.
Intuitively, smaller indexes need lower
$M$
values to reach the same target recall (e.g.,
’s
attr=C
requires lower
$M$
to serve queries at average
$x$
recall vs.
attr=D
,
), which we describe in
section

4.2
.

##### What Indexes to Build?

SIEVE
aims to build subindexes that efficiently serve (observed) queries with which alternative methods (e.g., brute-force KNN) are inefficient (i.e.,
marginal benefits
). Suppose we have the base HNSW index in
: While building subindex (
attr=D
) benefits its respective filtered query,
attr=D
has high selectivity (50%) that the base index serves it
fast enough
via result-set-filtering. In comparison, subindex (
attr=AorB
) is high marginal benefit: It serves (
attr=AorB
) significantly faster than the base index (
).
Subindexes can also serve non-exact matching filtered queries: For example, (
attr=AorB
) can also serve (
attr=A
) effectively, which has high-enough (50%) selectivity in the subindex. This expands utility of subindexes like (
attr=AorB
) from applicability to other filters. We describe
SIEVE
’s index selection in
section

4.3
.

##### How to Serve Queries?

SIEVE
decides between indexed search or brute-force KNN when serving queries with a built index collection. A key parameter controlling HNSW indexes’ search speed/recall tradeoff is the search exploration factor
$sef$
(
): higher
$sef$
(
over-searching
the graph) trades lower speed for higher recall (
).
SIEVE
will need to tune
$sef$
if the serving target recall is higher than that assumed at construction; Similar to
$M$
,
SIEVE
aims to use the
lowest
$sef$
for indexed searches, and smaller subindexes also require smaller
$sef$
for the same target (
). Then, given the best found index and
$sef$
,
SIEVE
evaluates whether falling back to brute-force KNN is faster (e.g.,
$sef\!>\!30$
,
), which also always has perfect recall. We describe
SIEVE
’s serving strategy in
section

5.2
.

## 3.SIEVE Framework Overview

SIEVE
(
) effectively serves filtered vector queries by building and using an index collection.
section

3.1
describes
SIEVE
’s index construction;
section

3.2
describes how
SIEVE
serves filtered vector queries.

### 3.1.SIEVE Construction

During construction,
SIEVE
aims to build a collection of the most beneficial HNSW subindexes given a memory budget and target recall based on the data distribution and a historical query workload.

##### Inputs

SIEVE
takes as input (1) an attributed vector dataset—a set of vectors and their scalar attributes, (2) a historical query workload—a set of query filters with probability/frequency counts, (3) a target recall, and (4) a memory budget. Unlike some specialized indexes (e.g., CAPS
(
Gupta et al. 2023
)
, HQANN
(
Wu et al. 2022
)
),
SIEVE
does not restrict attribute or filter forms, only requiring filters to be evaluable on attributes, e.g.,
A in attr
evaluates to
True
for
attr={A,B}
. (
section

4.1
)

##### Cost Modeling

SIEVE
models candidate indexes’ memory size and serving speed given their construction with sufficient density/
$M$
to serve queries at the target recall (
section

2.3
). It then accordingly sets up the candidates’ unit (marginal) benefits for optimization. (
section

4.2
)

##### Optimization

SIEVE
selects the subindexes to build under the memory budget with greedy submodular optimization, prioritizing high-unit marginal benefit and/or high (re)use-probability subindexes in a manner akin to
Materialized View Selection

(
Zhang et al. 2001
;
Shukla et al. 1998
;
Aouiche et al. 2006
)
. (
section

4.3
)

##### Indexing

SIEVE
builds the chosen subindexes over the dataset.
SIEVE
always includes the
base index
over the entire vector dataset in the collection, which acts as a fallback for queries that any other subindex in the collection cannot effectively handle. This design choice allows
SIEVE
to handle arbitrary (un-)filtered queries (
section

7
).

### 3.2.Serving Queries with SIEVE

For serving,
SIEVE
aims to choose the optimal search method for filtered queries based on the subindexes in the built collection and (a potentially different from construction-time) target recall.

##### Identifying the Optimal Indexed Search

The first, straightforward approach
SIEVE
uses for serving a query is with a built subindex.
SIEVE
finds the best subindex/
$sef$
combination for serving the query at target recall—following intuition in
section

2.3
, preferably a small subindex in which the query is dense, using low
$sef$
. (
section

5.1
)

##### Choosing Search Method

SIEVE
chooses between serving the query with the best-found subindex/
$sef$
combination (with result-set-filtering if needed) or brute-force KNN.
SIEVE
estimates serving speed of both methods with its cost model, then chooses the faster one, analogous to
MV-aware query rewriting

(
goo [n.d.]
;
Theodoratos and Xu 2004
;
Yang et al. 2018
)
. (
section

5.2
)

## 4.Index Collection Construction

This section covers how
SIEVE
builds its index collection. We describe preliminaries in
section

4.1
,
SIEVE
’s cost model and optimization problem (
SIEVE-Opt
) in
section

4.2
, and solution to
SIEVE-Opt
in
section

4.3
.

### 4.1.Preliminary and Definitions

###### Definition 4.1.

An
Attributed Dataset
is a set of
$n$
vectors
$\mathcal{V}=\{v_{1},...,v_{n}\}$
and a set of
$n$
attribute sets
$\mathcal{A}=\{a_{1},...,a_{n}\}$
, where each
$a_{i}$
is an attribute value set associated with each vector
$v_{i}\in\mathbb{R}^{d}$
.

depicts an example where each
$a_{i}$
is a set of strings.

###### Definition 4.2.

An
Filtered Query Workload
is pair of sets of
$m$
vectors and
$m$
filters
$\mathcal{M}=\{w_{1},...,w_{m}\}$
and
$\mathcal{F}=\{f_{1},...,f_{m}\}$
, where
$f_{i}$
is the filter of query vector
$w_{j}\in\mathbb{R}^{d}$
and each
$f_{i}:\mathcal{A}->\{0,1\}$
is a function that maps attribute values
$a_{j}\in\mathcal{A}$
to a binary indicator.

Each query filter
$f_{i}$
can be evaluated on the attributes
$a_{j}$
of each vector
$u_{j}$
:
$a_{j}$

satisfies

$f_{i}$
if
$f_{i}(a_{j})=1$
. For example, in
,
$f_{1}$
= (
A in attrs
) (shortened to
A
for brevity) evaluates to 1 on
$a_{1}$
=
{A,E}
, while
$f_{2}$
= (
D
$\land$
E
) evaluates to 1 on
$a_{2}$
=
{D,E}
.
3
3

3

SIEVE
defines attributes, filters, and evaluations following the RDBMS model
(
Ramasamy et al. 1998
)
, where attributes are column values that filters can be applied on, e.g.,
gender=female && price<20
is a filter evaluable on attributes
gender
and
price
from two different columns. We express attributes nevertheless as sets for simplicity in this paper.

We define the
cardinality
of each filter
$f_{i}$
as the number of dataset rows that satisfy the filter, i.e.,
$card(f_{i})=|\{a_{j}\in\mathcal{A}|f_{i}(a_{j})=1\}|$
.

###### Definition 4.3.

A
Filtered Vector Search Problem
takes an attributed dataset
$(\mathcal{V},\mathcal{A})$
, a filtered query workload
$(\mathcal{M},\mathcal{F})$
, and a similarity metric
$\mathcal{S}$
. The output is a
$|\mathcal{M}|\times k$
matrix
$\mathcal{R}$
of top-k results, where each row
$R_{i}=\{v_{i_{1}},...,v_{i_{k}}\}$
is the top-k closest vectors in
$\mathcal{V}$
based on
$\mathcal{S}$
that satisfy filter
$f_{i}$
, i.e.,
$f_{i}(a_{i_{l}})=1,\forall 1\leq l\leq k$
.

A solution
$\mathcal{R}$
’s quality is commonly evaluated via ①
recall
=
$\frac{|\mathcal{R}\cap\mathcal{R}^{*}|}{m\cdot k}$
where
$\mathcal{R}^{*}$
is the actual top-k, and ②
latency
=
$\frac{t}{m}$
or
Queries-per-second
(QPS,
$\frac{m}{t}$
), where
$t$
is the total search time. Effective filtered search can be achieved by serving queries with

(sub)indexes
(
section

2.3
):

###### Definition 4.4.

A
Subindex

$I_{f_{i}}$
is an index constructed over a subset of data points that satisfy filter
$f_{i}$
, i.e.,
$\mathcal{V}_{f_{i}}:=\{v_{j}|f_{i}(a_{j})=1\}$
.

For example, the subindex
$I_{\text{A}}$
only indexes the first three rows in
. The
base index
indexing all rows can be expressed as
$\mathcal{I}_{\mathcal{1}}$
, where
$\mathcal{1}$
is a ‘dummy filter’ that always evaluates to 1. Given a subindex
$I_{f_{i}}$
, it ① can be used to evaluate a filtered query
$(w_{j},f_{j})$
with serving cost
$C(I_{f_{i}},w_{j},f_{j})->\mathbb{R}+$
, and ② has a (in-memory) size
$S(I_{f_{i}})->\mathbb{R}+$
, for which we perform cost modeling in
section

4.2
.

###### Definition 4.5.

A
Historical Query Workload
is a query filter tally
$\mathcal{H}=\{(h_{1},c_{1}),...,(h_{l},c_{l})\}$
where filter
$h_{i}$
has occurred
$c_{i}$
times.

depicts a workload with 6 unique filters.
SIEVE
assumes
filter stability

(
Sun et al. 2014
;
Mohoney et al. 2023
)
for anticipated future workloads: the future workload’s filter distributions
$\mathcal{F}$
follow those observed in
$\mathcal{H}$
.

SIEVE
’s definitions only require all query filters to be evaluable on all dataset attributes, hence can inherently handle arbitrarily complex predicates and attributes. However, the specific predicate and attribute forms may affect
SIEVE
’s optimization quality (
section

6
) and other nuances such as adaptability to workload shifts (
section

7.7.2
).

### 4.2.SIEVE-Opt: Problem Setup

This section defines
SIEVE
’s problem: candidate subindexes for construction and their benefits/costs (i.e., index speed/size when serving queries at target recall), then formalizes
SIEVE-Opt
.

##### Candidate Subindex DAG

There are exponentially many possible subindexes for an attributed dataset.
Besides the base index,

SIEVE
limits its problem space by only considering subindexes corresponding to filters in
$\mathcal{H}$
.
For example, in
,
$I_{\text{A}}$
is a candidate while
$I_{\text{B}}$
is not.
The
YFCC
dataset, with 100K queries, produces 24K candidates
(
big-ann benchmarks 2024
)
. For optimization,
SIEVE
organizes candidates in a directed acyclic graph (DAG) where edges represent subsumption (e.g.,
$(I_{\text{A$\lor$B}},I_{\text{A}})$
in
),
4
4

4

SIEVE
currently defines and evaluates subsumption logically following established theoretical work
(
Gottlob 1987
)
. However, other definitions can potentially also be used (
section

6
).
which enables computing of subindex unit marginal benefits (described in
section

4.3
).

##### Defining Target Recall

Without loss of generality,
SIEVE
takes in a base
$M_{\infty}$
value for calibrating the query serving target recall, defined as the average recall of searching in the base index
$I_{\infty}$
built with
$M\!=\!M_{\infty}$
and
$sef\!=\!k$
, where
$k$
is the number of results to return and the lower bound of
$sef$
(i.e., no over-searching).

##### Indexing Parameters at Target Recall

SIEVE
aims to build subindexes with sufficient parameters to serve queries at target recall (
section

2.3
). Either
$M$
(for construction) or
$sef$
(for serving) can be tuned to achieve this; however, for construction,
SIEVE
assumes that all subindexes will use a uniform minimum
$sef=k$
and tunes only
$M$
: this is because
$sef=k$
is the lowest-recall and fastest search parameterization; if
SIEVE
’s subindexes (with sufficient
$M$
) serves queries at target recall with
$sef=k$
,
SIEVE
’s index collection can too; hence,
SIEVE
can then evaluate subindexes based on highest potential speedups. Versus
$M_{\infty}$
used for
$I_{\infty}$
, candidate subindexes
$S(I_{h})$
are evaluated and built with downscaled
$M$
(
):

###### Definition 4.6.

The
Subindex
$M$
downscaling function

$\mathcal{M}_{\downarrow}$
takes in a subindex
$I_{h}$
, and returns the
$M$
value required to build
$I_{h}$
with to achieve at least the same average query serving recall as the base index
$I_{\infty}$
built with
$M_{\infty}$
:
$\mathcal{M}_{\downarrow}(I_{h}):=\frac{M_{\infty}log(card(h))}{log(N)}$
.

SIEVE
’s intuition for
$\mathcal{M}_{\downarrow}$
is that HNSW graph layers (
section

2.1
) are based on Delaunay graphs
(
Dobkin et al. 1990
)
, which requires suitable node degrees (
$\Theta(logN)$
) for effective search. Hence, each subindex
$I_{h}$
’s
$M$
should match its size’s logarithm:
$\mathcal{M}_{\downarrow}(I_{h})\!\propto\!log(card(h))$
. For example, if
$I_{\infty}$
in
is built with
$M_{\infty}\!=\!32$
, subindex
$I_{\text{D}}$
would be built with
$M_{\text{D}}\!=\!\frac{32log(4)}{log(8)}\!\approx\!21$
.

SIEVE
will evaluate the memory size (explained shortly) of each candidate subindex
$I_{h}$
assuming construction with
$M\!=\!\mathcal{M}_{\downarrow}(I_{h})$
and
$sef\!=\!k$
(
$=1$
, for discussion).

##### Subindex Memory Size

Each subindex
$I_{h}$
has a memory size proportional to indexed points
$card(h)$
and
$M$
:
$S(I_{h})=M\cdot card(h)$
(
).
For example,
$I_{\text{D}}$
indexing
$4$
points built with
$\mathcal{M}_{\downarrow}(I_{h})=21$
has size
$84$
. Notably, due to
$M$
’s effect on memory,
SIEVE
’s
$M$
downscaling (
$\mathcal{M}_{\downarrow}$
) saves memory for smaller subindexes, enabling more subindexes to be built under the same memory constraint versus a naive method that builds all subindexes with a uniform
$M_{\infty}$
(
section

7.6
).

##### Subindex Search Cost

SIEVE
defines subindexes’ search costs as their serving latency:

###### Definition 4.7.

The
Indexed Search Cost Function
(with result-set-filtering,
section

2.2
)
$C$
takes a subindex
$I_{h}$
,
$sef$
, and a filtered query
$(w,f)$
, and returns the expected latency of using
$I_{h}$
with
$sef$
to serve
$(w,f)$
:
$C(I_{h},sef,w,f):=log(card(h))\cdot sef\cdot(\frac{card(h)}{card(f)})^{cor(w,f,h)}$
.

SIEVE
bases
$C$
on that HNSW’s search time scales logarithmically
(
Malkov and Yashunin 2018
)
with graph size and linearly with
$sef$

(
Malkov and Yashunin 2018
)
, and there is
$\frac{card(h)}{card(f)}$
probability that a data vector similar to the query vector
$w$
passes the filter with result-set-filtering (
section

2.2
), scaled by
query correlation
—
$cor(w,f,h)$
—ratio of average distance from
$w$
to points that satisfy
$f$
in
$I_{h}$
versus non-satisfying points
(
Patel et al. 2024
)
.
5
5

5

$M$
also potentially affects latency; however, there is no definite analytical nor empirical trend
(
Malkov and Yashunin 2018
;
Pinecone [n.d.]
)
(
section

7.6
), hence we omit it for simplicity.
Positive correlation (
$cor(w,f,h)<1$
) improves query performance, mitigating low selectivity’s effects as satisfying vectors are reached faster. Conversely, negative correlation (
$cor(w,f,h)>1$
) amplifies low selectivity’s impact and increases query cost.
SIEVE
assumes constant correlation across all subindexes and filters, i.e.,
$cor(w,f,h)==c$
, and for discussion, set
$c=1$
and simplify
$C(I_{h},sef,w,f)$
as
$C(I_{h},f)$
(as
$sef$
is also assumed to be fixed at
$1$
) in this section.
For example, in
, serving a query with filter
A
with
$I_{\text{A}\lor\text{B}}$
incurs
$\frac{4log(4)}{3}$
cost.
SIEVE
constrains for simplicity that a subindex
$I_{h}$
can only serve a query with filter
$f$
if
$h$
subsumes
$f$
; otherwise,
$C(I_{h},f)=\infty$
.
6
6

6

We study unconstrained cases, e.g., for multi-subindex search, in
section

A.1
.

##### Brute-force Search Cost

Any query
$(w,f)$
can be served via brute-force KNN, performing distance computations between
$w$
and all vectors in
$\{\mathcal{V},\mathcal{A}\}$
that satisfy
$f$
, i.e.,
$\mathcal{V}_{f}:=\{v_{i}|f(a_{i})=1\}$
. This trivially incurs cost
$C_{bf}(f)=card(f)$
linear to the cardinality.

##### Aligning Search Costs

The alignment between indexed and brute-force search costs is influenced by factors such as distance function implementation
(
CloudFlare [n.d.]
)
and index memory access patterns
(
Gao and Long 2023
)
. Hence,
SIEVE
scales the brute-force search cost
$C_{bf}$
with a constant
$\gamma\in\mathbb{R}+$
for alignment:
SIEVE
compares
$C$
with
$\gamma\cdot C_{bf}$
when evaluating indexed versus brute-force search. For illustration purposes, however, we assume
$\gamma\!=\!1$
. The aligned costs allow us to define the cost of the
best serving method
for a query
$(w,f)$
given an index collection
$\mathcal{I}$
:

###### Definition 4.8.

The collection query serving cost function
$C$
takes in a subindex collection
$\mathcal{I}:=I_{h_{1}},...,I_{h_{x}}$
and a filtered vector query
$(w,f)$
, and returns the cost of the
best possible serving strategy
given
$\mathcal{I}$
:
$C(\mathcal{I},f):=min(C_{bf}(f),min(\{C(I_{h},f)|I_{h}\in\mathcal{I}\})$
.

$C(\mathcal{I},f)$
represents the lower cost of ① brute-force KNN and ② searching with the smallest subindex subsuming
$(w,f)$
in
$\mathcal{I}$
: if
$\mathcal{I}$
is the entire DAG in
section

4.2
,
$C(\mathcal{I},\text{A})=log(3)$
as it is best served by its corresponding subindex
$I_{\text{A}}$
, while
$C(\mathcal{I},\text{F})=1$
, as its best indexed search (with
$I_{\infty}$
) costs
$8log(8)\div 1\approx 16.6$
, more than brute-force KNN (
$1$
). With the collection serving cost
$C(\mathcal{I},f)$
and index size
$S(I_{h})$
,
SIEVE
’s optimization problem,
SIEVE-Opt
, can be defined:

###### Problem 1.

SIEVE-Opt

Input:    :

(1):

Attributed Vector Dataset
$\{\mathcal{V},\mathcal{A}\}$

(2):

Historical Workload Distribution
$\mathcal{H}=\{(h_{i},c_{i})\}$

(3):

$M_{\infty}$
representing target recall

(4):

memory budget for subindex collection
$B$

Output::

(1):

A subindex collection to construct
$\mathcal{I}:=\{I_{h_{i_{1}}},...,I_{h_{i_{x}}}\}$

Objective function::

Minimize collection query serving cost over historical workload
$C(\mathcal{I},\mathcal{H})=\sum^{|\mathcal{H}|}_{i=1}c_{i}\cdot C(\mathcal{I},h_{i})$
.

Constraints::

:

Base index must exist:
$I_{\infty}\in\mathcal{I}$

:

Total subindex size is less than the memory budget
$\sum^{x}_{l=1}S(I_{h_{i_{l}}})\leq B$

##### Presence of the Base Index

SIEVE
enforces that
$I_{\infty}$
must exist for handling arbitrary (e.g., unseen) filtered queries. Worst case,
SIEVE
can serve queries with the better of
$I_{\infty}$
or brute-force KNN, which lower-bounds
SIEVE
’s serving performance (
section

7.1
).

### 4.3.SIEVE-Opt: Solution

This section presents our solution to
SIEVE-Opt
, whose formulation naturally gives rise to a greedy solution for subindex selection
(
Amanatidis et al. 2021
)
.

##### Marginal Benefits

Adding a new index
$I_{h}$
into
$\mathcal{I}$
decreases the collection query serving cost by its marginal benefit w.r.t.
$\mathcal{H}$
:
$C(\mathcal{I},\mathcal{H})-C(\mathcal{I}\cup\{I_{h}\},\mathcal{H})\geq 0~\forall\mathcal{I},\mathcal{H},I_{h}$
. For example, in
, if
$\mathcal{I}\!=\!\{I_{\text{A}\lor\text{B}},I_{\infty}\}$
,
$\mathcal{H}\!=\!\{(\text{A},1)\}$
(left), adding
$I_{\text{A}}$
into
$\mathcal{I}$
brings
$\frac{4log(4)}{3}\!-\!log(3)\approx 0.75$
marginal benefit. This is less (
$4.45$
) than adding
$I_{\text{A}}$
into a collection
$\mathcal{I}\!=\!\{I_{\infty}\}$
with only a base index (right)—there is
diminishing returns
with adding
$I_{\text{A}}$
when
$I_{\text{A}\lor\text{B}}$
exists. This property is generalizable:

|  | $$ \underbrace{C(\mathcal{I}\cup\{I_{h}\},\mathcal{H})-C(\mathcal{I},\mathcal{H})}_{\text{large marginal benefit}}\leq\underbrace{C(\mathcal{J}\cup\{I_{h}\},\mathcal{H})-C(\mathcal{J},\mathcal{H})}_{\text{small marginal benefit}}\forall\mathcal{I}\subseteq\mathcal{J} $$ |  |
| :--- | :--- | :--- |

That is, the query serving cost
$C(\mathcal{I},\mathcal{H})$
is a
supermodular set function
w.r.t
$\mathcal{I}$

(
Topkis 1998
)
, and
SIEVE-Opt
is a
supermodular minimization problem
with the knapsack memory constraint
$B$

(
Chekuri [n.d.]
)
.
This problem class gives rise to an empirically effective greedy algorithm—
GreedyRatio

(
Amanatidis et al. 2021
;
Mirzasoleiman et al. 2016
)
:
7
7

7

While theoretically bounded solutions exist
(
Sviridenko 2004
)
, their high overhead (e.g.,
$O(|\mathcal{H}|^{5})$
computations) makes them impractical
(
Mirzasoleiman et al. 2016
)
.
It starts with
$\mathcal{I}=\{I_{\infty}\}$
, then iteratively adds the highest marginal benefit/index size ratio subindex (
unit marginal benefit

$\frac{C(\mathcal{I}\cup\{I_{h}\},\mathcal{H})-C(\mathcal{I},\mathcal{H})}{S(I_{h})}$
) until reaching the constraint.

##### Example ()

Using
’s problem setting,
$M_{\infty}\!=\!10$
for
$I_{\infty}$
and
$\sum_{I_{h}\in\mathcal{I}}S(I_{h})\!<\!165=B$
,
GreedyRatio
proceeds as follows:

(1)

Step 1:

$I_{\text{A}}$
is selected (top right). Its unit benefit is high (
$0.253$
): serving
A
with
$I_{\text{A}}$
is much better than via brute-force search, and
$I_{\text{A}}$
is space efficient, requiring only
$\mathcal{M}_{\downarrow}(I_{\text{A}})=\frac{10log_{10}(3)}{log_{10}(10)}=5$
.

(2)

Step 2:

$I_{\text{D}}$
is selected (bottom left). Its unit benefit is high (
$0.217$
): serving
D
with
$I_{\text{D}}$
is much better than via the root index
$I_{\infty}$
.

(3)

Step 3:

$I_{\text{A}\lor\text{B}\lor\text{C}}$
is selected (bottom right). While its unit marginal benefit (0.209) is decreased by the already constructed
$I_{\text{A}}$
, it still has high marginal benefits for serving both
A
$\lor$
B
$\lor$
C
and
A
$\lor$
B
.

No index can be further added to
$\mathcal{I}$
without exceeding
$B$
, hence,
$\mathcal{I}=\{I_{\infty},I_{\text{A}},I_{\text{A}\lor\text{B}\lor\text{C}},I_{\text{D}}\}$
is the index collection that
SIEVE
constructs.

##### Analysis

GreedyRatio
has time complexity
$O(E+|\mathcal{H}|log(|\mathcal{H}|))$
where
$E$
is the candidate subindex DAG’s edge count (
), using a priority queue for sorting unit marginal benefits and after adding each subindex, updating its parents’ and children’s benefits. Optimization time is negligible versus
SIEVE
’s construction time: For example, on the
YFCC
dataset
(
big-ann benchmarks 2024
)
with 6,006 candidates to optimize over,
SIEVE
solves
SIEVE-Opt
in (only) 18ms, versus the 136 seconds for building the index collection post-optimization (
section

7.3
).

## 5.Query Serving

This section describes
SIEVE
’s dynamic query serving strategy with the built index collection and a (potentially different from construction-time) target recall.
SIEVE
first finds the optimal subindex for an incoming query (
Section

5.1
), then determines the optimal search method—parameterized index search or brute-force KNN (
Section

5.2
).

### 5.1.Identifying the Optimal Subindex

This section describes how
SIEVE
efficiently finds optimal subindexes for query serving.
SIEVE
’s cost model (
section

4.2
) dictates that a query
$(w,f)$
is best served with the smallest subindex
$I_{h}$
(i.e., minimum
$card(h)$
) in
$\mathcal{I}$
where the subindex filter
$h$
subsumes the query filter
$f$
, following uniform query correlation assumptions in
section

4.2
.

##### Index Collection DAG

Like the candidate DAG in
section

4.1
,
SIEVE
builds a DAG, specifically, a
Hasse diagram
(
Wikipedia [n.d.]
)
, over the index collection: given two subindexes
$I_{h},I_{q}\in\mathcal{I}$
, a directed edge
$(I_{h},I_{q})$
exists only if
$h$
subsumes
$q$
, and there is no other
$I_{u}\in\mathcal{I}$
such that
$h$
subsumes
$u$
, and
$u$
subsumes
$q$
.
(center) depicts the DAG built on the index collection from solving
SIEVE-Opt
in
.

##### DAG Traversal

For a filtered query
$(w,f)$
, the Index Collection DAG can be efficiently traversed via BFS starting from the root
$I_{\infty}$
to find the best subindex
$I_{h}$
: at each step, if the current subindex
$I_{q}$
’s filter does not subsume
$f$
, none of its descendants can either. In other words, for any descendant
$I_{p}$
of
$I_{q}$
in the DAG,
$p$
cannot subsume
$f$
, allowing the entire subgraph rooted at
$I_{q}$
to be pruned from the search.
For example, in
, the subindex
$I_{\text{A}\lor\text{B}\lor\text{C}}$
does not subsume the query filter
D
$\land$
(
C
$\lor$
E
), hence its child
$I_{\text{A}}$
can be skipped, efficiently leading to the best subindex
$I_{\text{D}}$
to be found. In practice, for the
YFCC
workload with 100K filtered queries and an index collection with 658 subindexes, finding the optimal subindex for all queries took (only) 297 ms, which is a low percentage of the total search time (e.g., minimum 20.76 seconds,
).

##### Remark

SIEVE
currently evaulates subsumptions for traversing the Hasse diagram logically (e.g.,
A
is subsumed by
A
$\lor$
B
). However, in cases where logical subsumption is rare (e.g., complex filter and attribute space,
UQV
dataset,
section

7.1
), other subsumption definitions such as bitvector-based subsumption can be used in its place (
section

6
).

### 5.2.Determining Optimal Search Strategy

This section outlines how
SIEVE
determines the serving method based on the best-found subindex. It first determines the search parameter (
$sef$
) required for the (new) target recall, then chooses between indexed search with the found
$sef$
or brute-force KNN.

##### Search Parameterization

Like
$\mathcal{M}_{\infty}$
(
section

4.2
), Users provide a
global

$sef_{\infty}$
to
SIEVE
(potentially different from the assumed build-time
$sef=k$
) for each query representing the serving-time target recall, defined as the expected recall of (over-)searching the base index
$I_{\infty}$
with
$sef_{\infty}$
. Following
section

2.3
,
SIEVE
aims to serve queries with
$sef$
values to match the target recall; hence, versus
$sef_{\infty}$
, lower
$sef$
(increments) can be used when serving queries with subindexes:

###### Definition 5.1.

The
Subindex
$sef$
downscaling function

$\mathcal{S}_{\downarrow}$
takes in a subindex
$I_{h}$
, and returns the
$sef$
value required to search
$I_{h}$
with to achieve at least the same average query serving recall as searching the base index
$I_{\infty}$
with
$sef_{\infty}$
:
$\mathcal{S}_{\downarrow}(I_{h}):=max(k,\frac{sef_{\infty}log(card(h))}{log(N)})$
, where
$k$
is the neighbors to query (and minimum value of
$sef$
,
section

4.2
) and assuming
$I_{h}$
was built with proportional
$M=\mathcal{M}_{\downarrow}(I_{h})$
.

SIEVE
’s intuition for
$\mathcal{S}_{\downarrow}$
is that each HNSW search visits
$O(logn)$
points
(
Malkov and Yashunin 2018
)
in its hierarchical structure (
section

2.1
): to maintain recall, the proportion between
$sef$
—the dynamic closest neighbors list size—and
$logn$
must be maintained, i.e., lists must cover a consistent proportion of the visited
$logn$
points, hence
$\mathcal{S}_{\downarrow}(I_{h})\propto log(card(h))$
. For example, if
$sef_{\infty}=50$
is specified for the base index in
as the serving-time recall, the same recall can be achieved with
$\mathcal{S}_{\downarrow}(I_{\text{D}})=\frac{50\times log(4)}{log(8)}=33$
when searching in the subindex
$I_{\text{D}}$
. As HNSW’s search time scales linearly with
$sef$

(
Malkov and Yashunin 2018
)
(
),
SIEVE
’s
$sef$
downscaling saves search time versus a static strategy that uses uniform
$sef_{\infty}$
for indexed searches while maintaining recall (
section

5.2
).

##### Indexed vs. Brute-force Search

Given a query
$(w,f)$
, its best subindex
$I_{h}$
in
section

5.1
, and downscaled
$sef_{h}=\mathcal{S}_{\downarrow}(I_{h})$
,
SIEVE
chooses between serving the query with indexed or brute-force KNN with its cost model from
section

4.2
: it chooses the lower-cost method out indexed search (
$C(I_{h},sef_{h},f)$
) and brute-force search (
$\gamma C_{bf}(f)$
). For example, in
, assuming
$k=1$
, serving
D
$\land$
(
C
$\lor$
E
) with
$I_{\text{D}}$
at
$sef_{\infty}=1$
has a lower cost
$max(1,\frac{1log(4)}{log(8)})\times\frac{4log(4)}{3}\approx 1.84$
vs. brute-force KNN (
$3$
), but at
$sef_{\infty}=3$
, brute-force KNN is faster as the indexed search cost becomes
$max(1,\frac{3log(4)}{log(8)})\times\frac{4log(4)}{3}\approx 3.670\geq 3$
.

## 6.Discussion

##### Size of Optimization Space

One potential problem for
SIEVE
is an exploding optimization space when the historical workload contains many distinct filter templates. To address this,
SIEVE
currently prunes small-cardinality candidates with no marginal benefit over brute-force KNN prior to solving
SIEVE-Opt
in
section

4.2
; if still insufficient,
SIEVE
can also only use top-
$k$
-common filters as candidates, which would often sufficiently approximate the full problem due to filter commonality
(
Sun et al. 2014
)
.
Large optimization spaces may also affect
SIEVE
during workload shifts, which we study in
section

7.7.2
.

##### Availability of Filter Cardinalities

SIEVE
assumes availability of accurate filter cardinality info (
$card(h)$
). This is because many recent vector search frameworks
(
Wei et al. 2020
;
Patel et al. 2024
;
nmslib 2024
;
Mohoney et al. 2023
;
Wang et al. 2021
)
separately manage scalar attributes using methods such as inverted indexes, B-trees, or partitioning. For search, filters will first be applied on scalars to compute a
bitmap
of passing vectors’ IDs (implying cardinality via nonzero count), which is then passed to the vector index.

##### Filter Evaluation Costs

While reported as part of total query serving time in experiments (
section

7
),
SIEVE
omits modeling of filter evaluation costs from optimization (
section

4
). This is because
SIEVE
currently follows the aforementioned bitmap-based filtering: for each query,
SIEVE
computes the bitmap before choosing the serving strategy (i.e., brute-force KNN or indexed search), hence its computation cost is orthogonal to
SIEVE
’s optimizations. Moreover, we find that bitmap computation time is negligible in our experiments; for example, on the
UQV
dataset, evaluating the complex, up to 10-attribute disjunction filters for 10K queries took (only) 16ms—0.2% of total query serving time at 0.95 recall (
).

##### Complex Predicate and Attribute Spaces

While
SIEVE
conceptually supports arbitrary predicates and attributes, a current limitation is that complex spaces (e.g., 200K attributes with up to 10-attribute disjunction filters of
UQV
,
) with few subsumption relations (2 random predicates rarely subsuming each other,
section

7.7.2
) can reduce subindex serving opportunities (for non-exact matching query and subindex filters) and performance (
section

5.1
). Potential mitigations are to use looser ① bitmap subsumption checks, where filter
B
subsumes
A
if all attributed vectors satisfying
A
also satisfy
B
even if logically otherwise, and ② expanded problem space with sub-predicates, such as considering
A
and
B
to also be valid candidates when
A
$\land$
B
is observed. Both methods increase chance of subsumptions between built subindexes and query filters while potentially increasing optimization time via costlier subsumption checks
(
Chambi et al. 2016
)
and larger problem space, respectively; a cost-based mechanism for choosing when to use them (e.g., Calcite’s
(
Begoli et al. 2018
)
rule-based usage of logical checks for simple inequalities) can be valuable future work.

##### Workload Shifts

We design
SIEVE
for production workloads with query filter stability
(
Sun et al. 2014
;
Sun et al. 2016
;
Idreos et al. 2007
;
Mohoney et al. 2023
)
where future queries can be predicted from past filters. Regardless, if filter distribution shifts from
$\mathcal{H}$
to a new
$\mathcal{H}^{\prime}$
,
SIEVE
can be incrementally updated by re-solving
SIEVE-Opt
over
$\mathcal{H}^{\prime}$
to find a new collection
$\mathcal{I}^{\prime}$
, building new indexes in
$\mathcal{I}^{\prime}\!-\!\mathcal{I}$
, then deleteing indexes in
$\mathcal{I}\!-\!\mathcal{I}^{\prime}$
(
section

7.7
). Notably, the base index
$I_{\infty}$
, which forms a significant part of
SIEVE
’s build time and memory size, does not need updating. Furthermore,
SIEVE
is robust to moderate shifts (
section

7.5
), and even for complete shifts (serving queries from unrelated
$\mathcal{H}^{\prime}$
when fit on
$\mathcal{H}$
),
SIEVE
’s performance will be lower bounded by
SIEVE
-NoExtraBudget (
section

7.7.2
).

##### Multi-Subindex Search

SIEVE
currently only considers serving queries with a single subindex that subsumes the query filter for indexed search (
section

4.2
). One potential alternative is to use multiple subindexes, e.g., re-ranking results from subindexes
$I_{p}$
and
$I_{q}$
to answer
$p\lor q$
. This can be useful for queries that
SIEVE
otherwise finds no good serving strategy (e.g., those with ’unhappy middle’ selectivities, but the best subindex found is the base index
$\mathcal{I}_{\infty}$
); However, finding optimal subindex sets for multi-index search is computationally hard. We evaluate the feasibility and potential gains of multi-subindex search in detail in our technical report
(
Zhaoheng et al. [n.d.]
)
.

## 7.Experiments

This section studies
SIEVE
’s performance on various filtered vector search workloads. We describe our experiment setup in
section

7.1
, study end-to-end query serving (
section

7.2
), index building (
section

7.3
), effect of memory budget (
section

7.4
) and historical workload (
section

7.5
) on index quality, our dynamic index building and serving parameterization (
section

7.6
), and
SIEVE
’s adapting to cold starts and workload shifts (
section

7.7
).

### 7.1.Experiment Setup

##### Datasets ()

① YFCC-10M
(
big-ann benchmarks 2024
)
: Dataset comes with queries with filters of form
A
or
A
$\land$
B
.
② Paper
(
Wang et al. 2022
)
: we generate data attributes where each vector has the
$i^{th}/20$
attribute with
$1/i$
probability as in NHQ
(
Wang et al. 2022
)
and Milvus
(
Wang et al. 2021
)
. Conjunctive AND query filters are generated following a zipf distribution
(
Sun et al. 2014
)
as described in HQI
(
Mohoney et al. 2023
)
.
③ UQV
(
UQV 2024
)
: we generate data/query filters following methodology of the
Paper
dataset, with
$1\!\leq\!i\!\leq\!200K$
and disjunctive OR filters.
④ GIST
(
Aude Oliva [n.d.]
)
: we generate 2 normally-distributed numerical attributes
$X$
and
$Y$
for each vector, and zipf-distributed disjunctive range filters.
⑤ SIFT
(
Texmex 2024
)
: we generate data/query filters following methodology of the
GIST
dataset with conjunctive range filters.
⑥ MSONG
(
Management and Preservation 2024
)
: we generate query filters uniformly of form
$a_{i}$
for 80% of queries; the remaining 20% are unfiltered.

##### Methods

We evaluate
SIEVE
against these existing methods:

(1)

ACORN-
$\gamma$

(
Patel et al. 2024
)
:

We use
$M\!=\!32$
,
$M_{\beta}\!=\!64$
, and
γ
=
m
a
x
(
80
,
1
/
\gamma\!=\!max(80,1/
min. filter selectivity
$)$
.
For each dataset, we sweep selectivity threshold for brute-force KNN fallback from 0.0005 to 0.05.

(2)

ACORN-1

(
Patel et al. 2024
)
:
Ablated
ACORN-
$\gamma$
with
$\gamma=1$
and
$M_{\beta}=32$
.

(3)

hnswlib

(
nmslib 2024
)
:
We use the better of
$M=\{16,32\}$
and
$efc=40$
.

(4)

SIEVE
-NoExtraBudget:
Ablated
SIEVE
with
$B=S(I_{\infty})$
that only builds a base index. Equivalent to
hnswlib
that falls back to brute-force KNN based on
SIEVE
’s serving strategy (
section

5.2
).

(5)

PreFilter
:

We first use the predicate filter, then perform brute-force KNN with
hnswlib
’s SIMD-enabled distance function.

(6)

Oracle
:
Exhaustive indexing method which
ACORN-
$\gamma$
aims to mimic
(
Patel et al. 2024
)
; it builds a subindex for every observed filter.
Oracle
is expected to outperform
SIEVE
but incur higher TTI and memory cost; we present it as a bound for
SIEVE
’s performance.

(7)

FilteredVamana

(
Gollapudi et al. 2023
)
only supports filters of form
A in attr
(or no filter); we compare against it on
MSONG
only. We build in-memory using DiskANNPy’s recommended parameters
(
DiskANN [n.d.]
)
.

(8)

CAPS

(
Gupta et al. 2023
)
only supports conjunctive attribute matching on under 256 data attributes; we compare against it on
Paper
/
MSONG
only. We sweep cluster count from 10-1000 on each dataset.

For
SIEVE
, we use
hnswlib

(
nmslib 2024
)
for our index collection. For each dataset, we sweep
$M_{\infty}\!=\!\{16,32\}$
and
$efc\!=\!40$
for the base index, with downscaled
$M$
and same
$efc$
for subindexes (
section

4.2
). Budget
$B$
is set to
3
$\times$

hnswlib
’s index size on the same dataset. The brute-force scaling constant
$\gamma$
is empirically chosen where
$\gamma\cdot c_{bf}(f)\!=c(I_{h},f)$
when
$card(f)\!=\!card(h)\!=\!1000$
, i.e., brute-force KNN and perfect-selectivity indexed search cost the same for a 1K-cardinality filtered query with
$sef\!=\!k$
(
section

4.2
). The query correlation factor
$q(w,f,h)$
is set to 0.5 (i.e., average positive correlation,
section

4.2
) for all
$w,f,h$
.

##### Index Fitting

We use the first 25% query slice (unless otherwise stated, e.g., in
section

7.5
and
section

7.7
) as the observed workload
$\mathcal{H}$
, then serve all queries (including the fitting slice) with the built index, following methodology in prior workload-aware indexing works
(
Mohoney et al. 2023
)
.

##### Measurement.

For
Oracle
,
hnswlib
,
SIEVE
-NoExtraBudget, we generate QPS-recall@10 curves with
$sef\!\in\![10,110]$
in steps of 10. For
SIEVE
, we use
$sef_{\infty}\!\in\![10,110]$
for the base index and downscale
$sef$
for subindexes (
section

5.2
). For
ACORN-
$\gamma$
,
ACORN-1
, we vary
$sef\!\in\![10,510]$
in steps of 50. For
CAPS
, we vary
$np\!\in\![3K,30K]$
in steps of 3K. For
FilteredVamana
, we vary
$L\!\in\![10,510]$
in steps of 50.

##### Environment

Experiments are run on an Ubuntu server with 2 AMD EPYC 7552 48-core Processors and 1TB RAM. We store datasets on local disk, build indexes in-memory with 96 threads, and run queries with 1 thread in Neurips’23 BigANN challenge’s environment
(
harsha simhadri 2024
)
reporting best-of-5 QPS. Our Github repository
(
Zhaoheng et al. 2025
)
contains our
SIEVE
implementation and experiment scripts.

### 7.2.High and Generalized Search Performance

This section studies
SIEVE
’s overall filtered vector search performance vs. existing baselines: we run each method on applicable datasets and compare their generated QPS-recall@10 curves.

reports results.
SIEVE
is the best-performing non-Oracle
approach on all 6 datasets, achieving up to 8.06
$\times$
speedup (
YFCC
) at 0.9 recall versus
ACORN-
$\gamma$
. Notably,
SIEVE
also achieves higher recalls (>0.99 in
SIEVE
vs. peaking at
$\sim$
0.95 in
ACORN-
$\gamma$
) on low-selectivity datasets (
YFCC
,
Paper
,
UQV
): while
ACORN-
$\gamma$
’s induced subgraphs degenerate for selective queries (
section

2.2
),
SIEVE
actually builds the (sub)indexes in which filters are dense for effective search.

##### Bounded Performance

SIEVE
-NoExtraBudget bounds
SIEVE
’s performance (
section

5.2
) when the best subindex
SIEVE
finds for any query is the base index
$I_{\infty}$
(e.g., workload shifts,
section

7.7.2
), and can only choose between searching with
$I_{\infty}$
or brute-force KNN. While
SIEVE
significantly outperforms
SIEVE
-NoExtraBudget (4.01
$\times$
QPS on
YFCC
at 0.95 recall), the latter is still effective in its own right, outperforming
ACORN-
$\gamma$
on 2/6 datasets (
YFCC
,
MSONG
).

##### High Generalizability

SIEVE
’s filter and attribute format-agnostic formulation (
section

4.1
) allows it to handle (1) any number of data attributes and (2) a wide range of predicate forms—conjunctions (
YFCC
,
Paper
), disjunctions (
UQV
), and range filters (
GIST
,
SIFT
), unlike
FilteredVamana
and
CAPS
, which struggle with large data attribute sets, disjunctions, and range queries. In addition,
SIEVE
still outperforms
CAPS
on
Paper
(10.61
$\times$
QPS @ 0.9 recall) and both
CAPS
and
FilteredVamana
on
MSONG
(2.29
$\times$
QPS @ 0.9 recall).

##### Handles Diverse Selectivities

We additionally study
SIEVE
’s per-query selectivity band performance in
section

A.2
.

### 7.3.Low Construction Overhead

This section studies
SIEVE
’s construction overhead. We fix
$B$
as 3
$\times$

hnswlib
’s index size on the same dataset (
section

7.1
), and compare
SIEVE
’s TTI and memory consumption vs. existing methods.

reports results.
SIEVE
adheres to its memory budget w.r.t.
hnswlib
: Notably, while the budget is 3
$\times$

hnswlib
’s index size,
SIEVE
’s actual memory consumption including datasets is
<
3
×
<\!3\times
, being as low as 1.29
$\times$
on
GIST
as the high-dimensional (960) vectors contribute significantly to memory size (3.84GB). The TTI increase is also
<
3
×
<\!3\times
(
1.68
×
1.68\times
on
YFCC
to
2.78
×
2.78\times
on
GIST
), as subindexes’s build time scales superlinearly with vector count
(
Malkov and Yashunin 2018
)
, hence larger indexes (e.g., base index) take more indexing time per vector. Versus
ACORN-
$\gamma$
, while
SIEVE
uses more memory (up to 1.96
$\times$
,
SIFT
), its TTI is a fraction of
ACORN-
$\gamma$
’s (
$0.8\%$
on
YFCC
to
$9.5\%$
on
SIFT
) as it avoids
ACORN-
$\gamma$
’s specialized graph building (
section

2.2
). This makes
SIEVE
desirable when TTI is the main constraint instead of memory (e.g. on-disk indexing).
Versus
Oracle
, which has potentially prohibitive size (84.2GB,
YFCC
),
SIEVE
performs competitively (
) using as little as
$3.0\%$
and
$4.6\%$
of
Oracle
’s TTI and memory (
UQV
).

### 7.4.Efficient Usage of Memory Budget

This section studies
SIEVE
’s performance vs. its indexing budget
$B$
, which we vary from 1
$\times$
size of
hnswlib
(equivalent to
SIEVE
-NoExtraBudget) to 5
$\times$
and report the QPS-Recall@10 curves, resource consumption, and search strategy breakdowns on
YFCC
.

reports results.
SIEVE
’s QPS-Recall@10 trade-off expectedly improves with more budget. However, contrasting the linear increase in TTI and memory, the improvement diminishes for each 1
$\times$
budget increase—the serving time decrease for 100K queries at 0.95 recall from 1
$\times$
to 2
$\times$
is 2.63
$\times$
, but only 1.22
$\times$
from 4
$\times$
to 5
$\times$
. This is because
SIEVE
prioritizes building high-benefit subindexes (
section

4.3
, also verified in
section

A.3
), which is reflected in its search strategy breakdown in
: the first budget increase from 1
$\times$
to 2
$\times$
focuses on building subindexes for queries with highest gains from being served by a subindex versus brute-force or base index search, decreasing the spent time of the 2 methods by 25.5s and 51.6s, respectively. Further budget increases yield smaller reductions in brute-force (<19.3s) or base index search (<10.4s) time.

### 7.5.Effective Fitting from Historical Workload

This section studies the impact of discrepancies between the historical workload
$\mathcal{H}$
used to build
SIEVE
and the actual workload. We vary the query slice size we use as the historical workload for
SIEVE
from 25% (our default for other experiments) to 100% on
YFCC
, and report the QPS-recall@10 of each fitted index collection.

reports results.
SIEVE
fit with 25% workload performs comparably (
$96\%$
QPS at 0.9 recall) versus theoretically optimized
SIEVE
fit with 100% workload despite (1) the 25% slice only containing 42% of unique filter templates and (2) the two fits being significantly different—170/711 indexes in
SIEVE
(25% WL) are absent in
SIEVE
(100% WL) and 141/682 vice versa, showcasing
SIEVE
’s robustness to moderate workload shifts (we study larger shifts in
section

7.7.2
). At intermediate values, while each 25% slice increases
SIEVE
’s choices from seeing more unique filters, performance increase is negligible.

### 7.6.Dynamic Construction & Serving

This section studies the effectiveness of
SIEVE
’s dynamic, recall-aware index construction (
section

4.2
) and serving parameterization (
section

5.2
). We compare
SIEVE
’s QPS-recall@10 curves with ablated versions of
SIEVE
—using static
$M=M_{\infty}$
for all subindexes and/or static
$sef=sef_{\infty}$
for all searches—on
UQV
and
Paper
with
$M_{\infty}=\{32,64\}$
.

We report the results in
. At high (0.985) recall,
SIEVE
achieves up to 1.60
$\times$
QPS increase versus
SIEVE
with no optimization and up to 1.09
$\times$
QPS increase with only one of dynamic subindex construction (
$M$
) or query serving (
$sef$
) (
UQV
,
$M_{\infty}=64$
). Interestingly, while the
$M_{\infty}=64$
setting is more performant than
$M_{\infty}=32$
on
SIFT
, the opposite holds for
UQV
; we hypothesize that this is due to intrinsic hardness difference of the datasets, i.e.,
$M_{\infty}=32$
suffices for the base index in
UQV
,
8
8

8

Recall that
$M_{\infty}$
is user-specified (
section

4.2
).
and further increasing
$M_{\infty}$
results in increased latency with negligible recall gains.

##### More Indexes Under Same Memory Constraint

SIEVE
’s recall-aware subindex construction downscales the
$M$
parameter of smaller indexes in the collection (
section

4.2
): this decreases the memory consumption of these small indexes (
), which results in more built indexes under the same memory constraint (
).

##### Fewer Distance Computations

SIEVE
’s dynamic search parameterization downscales
$sef$
when searching with smaller indexes (
section

5.2
). This increases smaller indexes’s search efficiency from incurring fewer distance computations (
), and reduces searches
SIEVE
falls back to serving via brute-force KNN on (
).

### 7.7.Handling Cold Starts and Workload Shifts

This section studies
SIEVE
’s robustness to cold starting with no historical workload (
section

7.7.1
) and sudden workload shifts (
section

7.7.2
).

#### 7.7.1.SIEVEcan Effectively Cold Start

We choose the
YFCC
dataset, temporally slice the 100K queries into 20 5K workload slices, then sequentially serve slices to
SIEVE
initialized with no historical workload (i.e.,
$\mathcal{H}\!=\!\emptyset$
) and an index collection with only the base index
$\mathcal{I}_{\infty}$
. Each slice
$H^{\prime}$
, after serving, is added to the historical workload (i.e.,
$\mathcal{H}\!:=\!\mathcal{H}\!\cup\!H^{\prime}$
) and
SIEVE
’s index collection is incrementally updated following procedures described in
section

6
. We study per-slice query performance and
SIEVE
’s update time between slices.

We report results in
. As observed in
, while
SIEVE
’s per-slice QPS is lower than the theoretically optimized
SIEVE
(100% WL) (
section

7.5
) on the first 2 slices—
SIEVE
only has the base index for the
$1^{st}$
and a suboptimal index collection fit to the first 5K queries for the
$2^{nd}$
,
SIEVE
effectively cold starts as it observes more slices, reaching 97% QPS of 100% WL by the
$3^{rd}$
slice. This is also seen in
SIEVE
’s update time (
) and built/deleted subindexes per update (
): while
SIEVE
’s first update takes significant time (111s)—it uses all budget to build 499 subindexes fit to the first slice, subsequent updates become increasingly faster due to
SIEVE
’s observed workload
$\mathcal{H}$
quickly approaching the true workload distribution: it builds and deletes fewer subindexes per update, with update time becoming sub-second after the
$15^{th}$
slice.

#### 7.7.2.SIEVEcan Adapt to Complete Workload Shifts

We choose the
GIST
,
Paper
and
UQV
datasets, on which we generate alternative workloads with different filter templates
9
9

9

We accomplish this via setting alternative seeds for randomized generation.
that follow similar distribution characteristics (i.e., average selectivity,
section

7.1
). We study the performance of serving the alternative workload on
SIEVE
fit on the alternative workload versus
SIEVE
fit on the original workload (i.e., to simulate a workload shift), and time for re-fitting
SIEVE
’s index collection from the original to the alternative workload.

We report results in
. As expected, serving a workload with
SIEVE
that significantly differs from the workload that
SIEVE
was fit on expectedly causes degradation, achieving only 92%, 71% and 49% QPS of an index collection fit with the corresponding workload on
GIST
,
Paper
and
UQV
, respectively (
,
,
).

##### Quantifying Degradation

While query filter templates in the original and alternative workloads almost completely differ for all datasets—the optimal index collections of the workloads share only 1 subindex on
Paper
and 0 on
GIST
and
UQV
(
), the degradation degree depends on the filter space complexity (
section

6
): two random filters on
GIST
are most likely to have a subsumption relation, followed by
UQV
, then
Paper
, as
GIST
only has 2 range-filtered attributes versus
Paper
and
UQV
’s 20 and 200K for attribute matching. Hence,
SIEVE
still finds significant opportunities for query serving with subindexes despite the workload shift on
GIST
(
) to achieve 2.77
$\times$
speedup versus
SIEVE
-NoExtraBudget, finds fewer on
Paper
(
) with 1.35
$\times$
speedup, and almost none on
UQV
(
) and degrades to only 1.03
$\times$
speedup. Degradation can also occur if the predicates are sparse by complexity (e.g.,
A<5 && B in attr && C like
$\textbackslash$
w+
). Hence, a current limitation of
SIEVE
(and other general workload-driven methods
(
Sun et al. 2014
;
Sun et al. 2016
;
Mohoney et al. 2023
)
) is that large predicate spaces (e.g.,
UQV
) are inherently more difficult for
SIEVE
’s optimization w.r.t. workload shifts. However,
SIEVE
can be updated upon detecting such degradation/shifts either incrementally as in
section

7.7.1
or completely refitting to the new workload—notably, even if no subindexes are kept on refit (
), refitting is still faster than complete rebuild as the base index does not need updating (
section

6
).

### 7.8.Experimentation Summary

We claim the following w.r.t.
SIEVE
’s experimental evaluations:

(1)

Effective and generalizable search:

SIEVE
handles arbitrary data attribute and query filter formats, achieving up to 8.06
$\times$
higher QPS at 0.9 recall@10 versus the next best alternative on low and high selectivity query workloads alike (
section

7.2
).

(2)

Low construction overhead:

SIEVE
operates within its memory budget, requiring only up to 2.15
$\times$
memory of
hnswlib
and just 1% of time-to-index (TTI) versus
ACORN-
$\gamma$

(
Patel et al. 2024
)
(
section

7.3
).

(3)

Effective usage of memory budget:

SIEVE
achieves large performance gains even with small budget (e.g., 2
$\times$
of
hnswlib
) as its modeling effectively prioritizes high-benefit subindexes (
section

7.4
).

(4)

Effective fitting from historical workload:

SIEVE
’s construction requires only modest knowledge of the workload distribution—an index collection fit from a 25% workload slice performs within 96% of a collection fit on the true distribution (
section

7.5
).

(5)

Effective recall-aware construction and serving:

SIEVE
’s recall-aware index (
$M$
) tuning and dynamic serving (
$sef$
) achieves up to 1.19
$\times$
higher QPS at 0.985 recall versus ablated, recall-agnostic
SIEVE
versions with static parameterization (
section

7.6
).

(6)

Handling cold starts and workload shifts
: We show that
SIEVE
can handle cold start scenarios with no workload knowledge and complete workload shifts via incremental refitting (
section

7.7
).

## 8.Related Work

##### Filtered Vector Search

There exists many filtered vector search indexes
(
Patel et al. 2024
;
Gupta et al. 2023
;
Douze et al. 2024
;
Gollapudi et al. 2023
;
Wu et al. 2022
;
Mohoney et al. 2023
;
Wang et al. 2022
;
Landrum et al. 2025
;
Zuo et al. 2024
;
Engels et al. [n.d.]
;
Sanca and Ailamaki 2024
)
.

FilteredVamana
and StitchedVamana
(
Gollapudi et al. 2023
)
are Vamana-based
(
Jayaram Subramanya et al. 2019
)
indexes for single-attribute filters. CAPS
(
Gupta et al. 2023
)
and HQI
(
Mohoney et al. 2023
)
partition data to maximize query-time partition skipping; the latter is also workload-aware. However, CAPS only handles low-cardinality conjunctions, while HQI targets batched query serving.
NHQ
(
Wang et al. 2022
)
and HQANN
(
Wu et al. 2022
)
jointly indexing vectors and attributes but only support soft filters. SeRF
(
Zuo et al. 2024
)
and IVF2
(
Landrum et al. 2025
)
are specialized indexes for 1-attribute range queries
10
10

10

Hence, we omit SeRF from our experiments as it is not applicable.
and
YFCC
,
11
11

11

IVF2 was tuned specifically for
YFCC
, handling only filtered queries of form
a (AND b) in attr
. We omit it from experimentation due to its lack of generality.
respectively.
ACORN
(
Patel et al. 2024
)
supports arbitrary predicates via subgraph traversal.

SIEVE
supports arbitrary predicates like ACORN, but achieves higher QPS/recall with a faster-to-build, workload-aware index collection (
section

7.3
).
Versus SeRF, which also conceptually builds multiple subindexes,
SIEVE
focuses on its cost modeling for subindex selection in a potentially large, multi-attribute space (
section

4.2
); SeRF focuses orthogonally on compressing its subindexes that are otherwise naively and exhaustively built over a single attribute.
SIEVE
still outperforms other specialized methods (e.g.,
CAPS
) supporting limited predicates (
section

7.2
).

##### Vector Search Plan Selection

Cost-based query plan selection has been studied in vector indexing systems
(
Wang et al. 2021
;
Wei et al. 2020
)
.
Milvus
(
Wang et al. 2021
)
uses a cost model to choose between partitioned and pre-filter search. AnalyticDB-V
(
Wei et al. 2020
)
additionally considers query selectivity. In comparison,
SIEVE
dynamically picks the best strategy
considering recall
, unlike these systems (and ACORN’s resorting to brute-force KNN at low selectivity
(
Patel et al. 2024
)
) treating indexes as already tuned to serve with sufficient recall, and uses recall-agnostic models at serving time.
SIEVE
’s strategy
bounds
its performance: it will always be faster than the best of brute-force/indexed search at any recall (
section

7.2
).

##### Materialized View Selection

There is extensive work on selecting MVs for speeding up (exact) queries
(
Ivanova et al. 2010
;
Roy et al. 2019
;
Jindal et al. 2018
;
Mistry et al. 2001
;
Katsifodimos et al. 2012
;
Li et al. 2023
)
, which typically assume access to historical info for predicting the future workload
(
Sun et al. 2014
;
Sun et al. 2016
;
Mohoney et al. 2023
)
.
One common issue is the large candidate MV optimization space; BigSubs
(
Jindal et al. 2018
)
mitigates this via randomized approximation, while SparkCruise
(
Roy et al. 2019
)
reduces the problem space according to subsumptions.
While
SIEVE
shares similarities with these works such as fitting from historical workload (
section

4.1
) and using greedy approximation (
section

4.3
), it orthogonally incorporates recall (
section

4.2
), a key and unique dimension present in vector indexing for filtered search but lacking in MV selection for (exact) queries.

## 9.Conclusion

We present
SIEVE
, an indexing framework enabling efficient filtered vector search via an index collection.
SIEVE
uses a three-dimensional cost model for memory size, search speed, and recall to determine benefits and costs of candidate indexes at a target recall to build the index collection with bounded memory via workload-driven optimization. For query serving,
SIEVE
finds the fastest search strategy—a parameterized index search or brute-force KNN, at a potentially new target recall.
SIEVE
achieves up to 8.06
$\times$
QPS gain over existing indexing methods at 0.9 recall on various datasets while requiring as little as 1% TTI versus other specialized indexes.

## Appendix AAppendix

### A.1.Multi-index Search

This section studies the feasibility of serving filtered queries with multiple subindexes which when unioned, cover the query filter, as discussed in
section

6
. While
SIEVE
’s current serving strategy considers only (single) subindex search vs. brute-force KNN (
section

5
), it can be extended to multi-subindex search—given a filtered query
$w,f$
and index collection
$\mathcal{I}$
,
SIEVE
aims to choose between these options:

(1)

Single-index search:
$argmin_{I_{h}\in\mathcal{I}}C(I_{h},sef_{\downarrow}(I_{h}),w,f)$

(2)

Brute-force KNN:
$C_{bf}(f)=card(f)$

(3)

Multi-index search:
a
r
g
m
i
n
I
′
⊆
ℐ
∑
I
h
∈
I
′
C
(
I
h
,
s
e
f
↓
(
I
h
)
,
w
,
f
)
argmin_{I^{\prime}\subseteq\mathcal{I}\sum_{I_{h}\in I^{\prime}}}C(I_{h},sef_{\downarrow}(I_{h}),w,f)
such that
$f\subseteq\bigcup{h}_{I_{h}\in I^{\prime}}$
12
12

12

We omit re-ranking time as we find it to be negligible (e.g., contributes to
$\sim 0.1\%$
of total query time when re-ranking results from up to 10 subindexes).

Where
$I^{\prime}$
is the subset of indexes for multi-index search. As (only) the union of subindexes needs to cover the query filter, the constraint that each individual subindex
$I_{h}$
must cover
$f$
in the cost function
$C$
is lifted (i.e., as opposed to the original definition in
section

4.2
). Another notable difference is that to evaluate
$C(I_{h},f)$
for each
$I_{h}$
in
$I^{\prime}$
,
SIEVE
must estimate the
conditional selectivity
of
$f$
in
$I_{h}$
(e.g.,
$sel(1\!\leq\!A\!\leq\!5|0\!\leq\!A\!\leq\!3)$
, discussed in more detail shortly).

##### Potential Benefits

Based on
SIEVE
’s cost model (
section

4.2
), multi-index search can be beneficial when the query filter
$f$
is almost disjointly and exactly covered by a few subindexes
$I^{\prime}\subseteq\mathcal{I}$
, that is:

(1)

$\forall I_{h},I_{k}\in I^{\prime},|h\cap k|$
is minimized.

(2)

|
⋃
h
,
I
h
∈
I
′
−
f
|
|\bigcup_{h,I_{h}\in I^{\prime}}-f|
is minimized.

(3)

$|I^{\prime}|$
is minimized.

This is because (1) we want to avoid duplicately searching satisfying vectors in covering subindexes, (2) maximize conditional selectivity of the query filter in covering subindexes, and (3) as HNSW graphs’s search time is logarithmic (i.e., sub-linear) w.r.t. vector count, we aim to use few, large indexes as opposed many, small indexes.
The query must also be difficult to serve with alternative methods, i.e., it having low selectivity in the smallest single subindex that covers it (potentially the base index
$\mathcal{I}_{\infty}$
), but still having high enough cardinality such that brute-force KNN is also expensive.

##### Motivating Example ()

Given the query filter
$1\!\leq\!A\!\leq\!5$
, a near-optimal scenario for multi-index serving is using the two subindexes
$1\!\leq\!A\!\leq\!3$
and
$3\!\leq\!A\!\leq\!5$
which exactly and disjointly cover
$1\!\leq\!A\!\leq\!5$
. At the same time, the smallest (single) subindex that covers
$1\!\leq\!A\!\leq\!5$
is
$1\!\leq\!A\!\leq\!7$
, for which a (moderately selective) filtered search would have higher cost than the multi-index search following our cost model in
section

4.2
(
).
However, if the two covering subindexes overlap (
), contain non-satisfying vectors (
), or if we instead had to use three exactly covering and disjoint subindexes (
), the cost of the best multi-index search would become higher than that of single index search. At the same time, the existence of a smaller single index that covers the query filter with high selectivity (e.g.,
$1\!\leq\!A\!\leq\!6$
,
) can also potentially render multi-index search (relatively) sub-optimal.

##### Quantitative Evaluation ()

We evaluate the scenarios discussed in
on a test dataset with 1M 16-dimension random vectors and a query that matches 100K vectors (
). As hypothesized in
according to
SIEVE
’s cost model, adjusting each factor—increasing subindex count (
), increasing overlap between subindexes (
), and decreasing (conditional) selectivity of the query in the subindexes (
) all increase the latency of multi-index search. We also observe in
that multi-index search with 2 disjoint, exactly covering subindexes (
$I_{h_{1}}$
and
$I_{h_{2}}$
) is only beneficial if the query’s selectivity in the (single) covering subindex
$I_{c}$
is less than 0.7. Furthermore, if the query’s selectivity in
$I_{c}$
is 1 (i.e.,
$c_{r}=100K$
and
$I_{c}$
exactly matches the filter), serving with
$I_{c}$
becomes more optimal versus any possible multi-index search.
13
13

13

According to empirical results and
SIEVE
’s cost model; we omit the proof for brevity.

##### Difficulty of Multi-Index Search

Notably, the problem of finding the best union of subindexes
$I^{\prime}$
for a multi-subindex search is NP-hard (equivalent to weighted set cover where each candidate subindex
$I_{h}$
is weighted by query serving cost
$C(I_{h},f)$
), necessitating efficient greedy algorithms in cases where
SIEVE
has non trivially-sized index collections
(
Young 2008
)
. Furthermore,
SIEVE
must also make repeated evaluations of query filter
$f$
’s conditional selectivity in candidate subindexes
$I_{h}$
(e.g., the aforementioned
$sel(1\!\leq\!A\!\leq\!5|0\!\leq\!A\!\leq\!3)$
) for cost estimation. Versus cases where
$I_{h}$
subsumes
$f$
(i.e., for single-index search) where
$f$
’s selectivity in
$I_{h}$
can be trivially computed as
$\frac{card(f)}{card(h)}$
, conditional selectivities can potentially involve more expensive, data dependent computations.

##### Testing Multi-Index Search

We study the potential benefits versus costs of multi-index search in more detail by augmenting
SIEVE
to allow it to choose multi-index search for query serving (when more optimal versus both single-index search and brute-force KNN) following multi-index covers found with the greedy algorithm for weighted set cover
(
Young 2008
)
. We evaluate the augmented
SIEVE
’s performance versus
SIEVE
with no such augmentation on the disjunction-filtered datasets
UQV
and
GIST
.
14
14

14

YFCC
,
Paper
,
SIFT
, and
MSONG
contain (almost) no opportunities for multi-index search as their filters are conjunction-based: using attribute matching as an example, to cover
A
, one must union
A and B
,
A and C
,…, i.e., (many) subindexes with conjunctions between A and all other possible attributes in the attribute/filter space.

We report results in
. The query time between
SIEVE
with and without multi-index search enabled is negligible on
GIST
at <1% difference at 0.98 recall on
GIST
(
): this is because due to the various limiting factors discussed in
, we find that multi-index is rarely the optimal choice versus single-index search and brute-force KNN, being optimal for only 4/1000 queries (
), all of which union only 2 subindexes.
Additionally, while the overhead for finding near-optimal multi-index search strategies is negligible on
GIST
, it incurs prohibitive overhead on
UQV
, reducing the QPS @ 0.98 recall by more than
200
×
200\times
(
); this is because the greedy algorithm for weighted set cover has time complexity
$O(mn)$

(
Young 2008
)
where
$m$
is the candidate count (i.e, cardinality of
SIEVE
’s index collection
$|\mathcal{I}|$
, which is 14 on
GIST
and 200 on
UQV
) and
$n$
is the size of the attribute/filter space, of which
UQV
’s is significantly larger than
GIST
’s (
).
Hence, while multi-index search may bring potential benefits on workloads with sparse attribute/filter spaces such as
GIST
and can be a valuable extension for
SIEVE
, we defer exploration of low-overhead implementations of this technique and its extensions (e.g., additionally considering multi-index search for cost modeling during construction time,
section

4.2
) to future work.

### A.2.SIEVE’s Performance vs. Selectivity Band

reports
SIEVE
’s performance vs. query selectivity bands on
MSONG
. As theorized in
section

2.3
,
SIEVE
’s building of subindexes for “unhappy-middle“ queries (verified in
) provide large performance gains—2.00
$\times$
and 4.48
$\times$
speedup vs.
ACORN-
$\gamma$
and
hnswlib
at 0.99 recall—on the lowest band (
).
SIEVE
also deprioritizes optimizing for high-selectivity queries as it has limited budget; it simply routes them to the base index, performing the same as
hnswlib
(
,
). Interestingly,
ACORN-
$\gamma$
’s neighbor expansion is
detrimental
at high selectivity, incurring unnecessary overhead;
SIEVE
(and
hnswlib
) outperforms it by 2.36
$\times$
at 0.99 recall on the highest band (
).

### A.3.Distribution ofSIEVE’s Built Subindexes

presents the distribution of
SIEVE
’s built vs. candidate (i.e., not built) subindexes for the
YFCC
and
Paper
datasets at 3
$\times$
budget: following
SIEVE
’s intuition in
section

2.3
and cost model in
section

4.2
,
SIEVE
prioritizes subindexes with medium (
unhappy middle
) selectivity filters and/or high historical occurrences: Serving applicable queries with smaller subindexes bring limited benefits versus with brute-force KNN, while larger (non-base) subindexes provide only marginal improvements over serving with the base index.