# Overlap and transfer between commuting and attribute-specific urban origin–destination matrices

## Abstract

Open urban mobility products publish origin–destination (OD) count matrices on a shared tessellation, cut by trip purpose, time of day, calendar day, sex, age, trip length, and sometimes mode, income or place of residence. Downstream pipelines still treat morning commuting flows as the city’s destination field: they reuse the commuting table, its transpose, or a kernel fitted only on commuting. Global products now write that kernel at scale. WorldMove conditions synthetic trajectories on commuting OD; Rong et al. release generated intra-city commuting matrices for thousands of cities.

The object here is an \(N\times N\) trip-count matrix on one tessellation per study area. Condition S compares each published cut with morning home-based work (HBW) after the reference is scaled to the target total. Condition G fits named generators on HBW only and emits each target from that target’s observed origin outflows. The score is the common part of commuters (CPC), with Pearson correlation, NRMSE and Jensen–Shannon divergence as companions. Zero-shot generators are gravity (power and exponential, including a closed-form gravity-like law), radiation and Deep Gravity. Few-shot adaptation uses neuroGravity: a meta-Gravity prior whose \(G\) and \(\alpha\) are predicted from regional features, followed by an edge-enhanced graph transformer that learns residuals from a sparse sample of target edges.

## Introduction

Flow generation is the task of writing an origin–destination count matrix on a named tessellation when some or all of the target cells are unobserved. Luca et al. separate it from next-location prediction, crowd-flow forecasting and trajectory generation; the default overlap score for the matrix task is CPC. The zero-shot generators still named as baselines are a gravity law with power or exponential deterrence, the radiation law, and Deep Gravity, run production-constrained (\(\widehat{T}_{ij}=O_i\,p_{j\mid i}\)) and tested with spatial hold-out of regions or origins.

The matrix those papers generate is, in practice, a commuting matrix: census journey-to-work tables, or operator flows anchored on home and work. The same layer is then reused far outside that training cut. WorldMove conditions synthetic city-scale trajectories on commuting OD. Rong et al. publish a global intra-city commuting product for 2,358 cities. Activity-based models, time-dependent OD recovery, contact matrices and emission inventories attach non-work, off-peak or subgroup demand to the same skeleton. Vo et al. reconstruct mode-by-hour urban cubes from smart-card and survey fusion and score them with CPC; the cuts exist, but the commuting kernel is not the comparator.

Return-to-home and a stable radius of gyration are properties of trajectories. Two matrices on the same zones can share those regularities and still send trips to different destinations. Closed-form gravity-like laws remain competitive with high-capacity models on commuting tables. The open question is whether that commuting assignment, or a generator fitted only on it, recovers the other published cuts of the same tessellation.

Zero-shot generation (Deep Gravity, classical gravity, radiation) forbids using target-partition cells as input. Few-shot reconstruction is now a second published setting. neuroGravity initialises each edge with meta-Gravity—an MLP that predicts the gravitational constant and the distance-decay exponent from origin and destination features—then trains an edge-enhanced graph transformer on a sparse set of observed edges. Yang et al. report that about 1% of internal pairs reconstruct a commuting network at \(R^2=0.77\) in Boston and transfer zero-shot to other U.S. cities. We use the same stack as the few-shot baseline for attribute transfer: meta-Gravity and the transformer see morning HBW, then a declared fraction of target-partition edges, and are scored with CPC on the held-out target cells.

## Data

### Object and tessellation

The object is an \(N\times N\) trip-count matrix on one tessellation per study area. Cell \(T_{ij}\) is the estimated number of trips from zone \(i\) to zone \(j\) in a stated partition. Intra-zonal cells are kept; their share \(h_{\mathrm{intra}}=\sum_i T_{ii}/\sum_{ij}T_{ij}\) is reported on the unsliced weekday total and on every scored partition. External and foreign centroids (MITMA NUTS-3 France/Portugal and the residual “rest of world” zone; Lombardy zones outside the chosen urban clip) are dropped before \(N\) is counted.

Inclusion, applied to the unsliced weekday total, not to a thin slice: \(N\ge 25\) and \(h_{\mathrm{intra}}\le 0.70\). Zones with zero outflow on that total are kept if they have inflow, so that column support is not silently trimmed.

Distance \(d_{ij}\) used by gravity, radiation and Deep Gravity is the Euclidean distance between official zone centroids in a metric CRS (ETRS89 / UTM zone of the study area; Monte Mario / Italy zone for Lombardy). Intra-zonal distance is the radius of a circle with the same area as the polygon, \(d_{ii}=\sqrt{A_i/\pi}\). CPCd uses the same Euclidean length unless a network length is named in the table caption.

### Products

**MITMA / MITMS Estudio de movilidad de viajeros, v2 (2022–).** National open OD built from Orange España signalling, calibrated to the resident population of Spain (methodological report v8). We use the district-level daily trip files, not the municipality or large-urban-area aggregates, unless a candidate area fails the intra-zonal cap at district resolution, in which case the official municipal dissolve is the fallback tessellation and is flagged in Table 1.

Each row is one combination of date, hour \(\{0,\ldots,23\}\), origin id, destination id, distance band, origin activity, destination activity, study-possible flags, province of residence, income band, age band, sex, with mass `n_trips` and `trips_total_length_km`. Vendor factor levels, kept as released:

- distance: \(0.5\)–\(2\), \(2\)–\(10\), \(10\)–\(50\), \(>50\) km (trips shorter than \(500\,\mathrm{m}\) are absent);
- activity: `home` / `casa`, `work_or_study` / `trabajo_estudio`, `frequent_activity` / `frecuente`, `infrequent_activity` / `no_frecuente`;
- study flags: `estudio_origen_posible`, `estudio_destino_posible`;
- age: \(0\)–\(25\), \(25\)–\(45\), \(45\)–\(65\), \(65\)–\(100\), plus a residual NA band that is never used as a scored partition;
- sex: female / male (NA dropped from sex partitions, kept in the unsliced total);
- income: household income \(<10\), \(10\)–\(15\), \(>15\) thousand euros;
- residence: INE province code.

National coverage is about \(3{,}792\) districts inside Spain plus foreign NUTS-3 dummies. Study areas are contiguous district clusters that correspond to official large urban areas or to a named metropolitan clip (for example a province plus adjacent commuter districts). The clip, the date window, and the list of valid dates after published outages are recorded in Table 1 and Supplementary Table S1. Default analysis window: weekdays in a school-term month after 2022 that the product marks as valid, excluding national holidays. Weekend matrices enter only the day-of-week partitions.

**Regione Lombardia Matrice OD2016 passeggeri.** One average weekday on \(1{,}525\) zones (\(1{,}450\) internal). Flows are published already crossed by time band (`FASCIA_ORARIA`), motive and main mode. Motives: work (`lavoro`), study (`studio`), occasional (`occasionali`), business (`affari`), home-bound (`rientri a casa`). Modes: car driver, car passenger, bus, rail, motorcycle, bicycle, walk, other. The study-area clip is the set of internal zones whose centroids fall in the Milan functional urban area or an official provincial subset with \(N\ge 25\); external gates are dropped. There is no calendar stack, so Condition-S day-level bootstrap is not defined for Lombardy.

**Other planning or household-survey matrices.** Admitted only when purpose and start period exist on a public tessellation that can be matched to population and OSM. One representative weekday. Motive and period maps follow the same coarse alphabet as MITMA; school is study.

Population and jobs for attractiveness: INE census-section population and, where published, employment, aggregated to MITMA districts with the official crosswalk; ISTAT / Regione Lombardia socio-economic attributes attached to OD2016 zones. OSM features for Deep Gravity are extracted from a dated planet extract clipped to a \(5\,\mathrm{km}\) buffer around the study area (Methods).

Study areas that pass inclusion, with \(N\), \(h_{\mathrm{intra}}\), days and attributes present, are listed in Table 1.

### Reference matrix and codebook

\(\mathbf{R}\) is morning home-based work on the study-area tessellation.

MITMA construction. Restrict to weekdays in the analysis window. Keep hours in the AM bin (default \(07\)–\(10\), i.e. vendor hours \(7,8,9\); endpoints in S1 if a local peak file exists). Keep rows with `activity_origin = home` and `activity_destination = work_or_study`. If `estudio_destino_posible` is true, those rows are excluded from \(\mathbf{R}\) and stored as the study partition. Sum `n_trips` over remaining attributes onto \((i,j)\). The same hours and origin activity with destination `home` are not \(\mathbf{R}\); they are home-bound AM.

Lombardy construction. Map `lavoro` in the morning time band to HBW. `studio` is always held out. `rientri a casa` in the evening band is home-bound PM.

Period collapse, both products, defaults unless S1 overrides:

| Bin | MITMA hours | Role |
| --- | --- | --- |
| AM | 07–10 | reference window; other-purpose AM; home-bound AM |
| PM | 16–20 | home-bound PM and residual PM slices |
| night | 20–24 | night slices |
| overnight | 00–07 | overnight slices |

Hours \(10\)–\(16\) are inter-peak. They are not merged into AM or PM. They may appear in Table 5 as an extra published period, not as a primary cell.

Activity collapse for Table 2: `frequent_activity` and `infrequent_activity` merge to `other`. They stay distinct in Table 5. Origin activity \(\times\) destination activity matrices in Table 5 are the four-by-four vendor alphabet, not the collapsed three-letter alphabet.

Length, age and income bins are never re-cut.

## Observed partitions versus morning commuting

### Condition S

Let \(\mathbf{T}\) and \(\mathbf{R}\) be observed count matrices on the same tessellation. Totals generally differ. Set
\[
\lambda=\frac{\sum_{ij}T_{ij}}{\sum_{ij}R_{ij}},\qquad \mathbf{R}^{\downarrow}=\lambda\mathbf{R}.
\]
Scores are computed on \((\mathbf{T},\mathbf{R}^{\downarrow})\). After totals match, CPC is the share of trips assigned to the correct destination in a production-constrained model.

When the working hypothesis is that \(\mathbf{T}\) reverses commuting, the same scores are computed on \((\mathbf{T},\lambda\mathbf{R}^{\top})\). The contrast
\[
\Delta=\mathrm{CPC}(\mathbf{T},\lambda\mathbf{R}^{\top})-\mathrm{CPC}(\mathbf{T},\lambda\mathbf{R})
\]
is positive when the target overlaps the transposed commuting table more than the commuting table.

Nulls under Condition S: mass-scaled reference \(\lambda\mathbf{R}\); mass-scaled transpose \(\lambda\mathbf{R}^{\top}\); independence table \(O_i^{T}D_j^{T}/\sum D^{T}\).

### Measures

**CPC**
\[
\mathrm{CPC}(\mathbf{A},\mathbf{B})=\frac{2\sum_{ij}\min(A_{ij},B_{ij})}{\sum_{ij}A_{ij}+\sum_{ij}B_{ij}}.
\]

Pearson correlation on \(\log(1+A_{ij})\) and \(\log(1+B_{ij})\) over cells with \(A_{ij}+B_{ij}>0\), reported as \(r\) and \(R^{2}=r^{2}\); NRMSE with denominator equal to the mean cell of \(\mathbf{A}\); Jensen–Shannon divergence of the normalised histograms of cell masses. CPL is Sørensen–Dice on binary support. CPCd is Sørensen–Dice on the binned Euclidean (or network; stated) trip-length distribution. When neuroGravity is scored, raw-flow \(R^{2}\) is stored next to CPC.

### Pre-specified partitions

1. Purpose \(\times\) time of day, coarse alphabet \(\{\mathrm{work},\mathrm{home},\mathrm{other}\}\times\{\mathrm{AM},\mathrm{PM},\mathrm{night},\mathrm{overnight}\}\). Primary targets against \(\mathbf{R}\): other-purpose AM; home-bound PM; home-bound AM.
2. Day of week or calendar day; sex; age band.
3. Trip-length bands as released (mobile-network bins \(0.5\)–\(2\), \(2\)–\(10\), \(10\)–\(50\), \(>50\,\mathrm{km}\)). Cells outside the band are zero.
4. Frequent versus infrequent non-home, non-work activity; study versus work when flagged; origin activity \(\times\) destination activity.
5. Mode, income band, province of residence, when published. Tourism-dominated areas: resident versus non-resident.

Day, sex and age partitions are expected to stay close to \(\mathbf{R}\) (high CPC, \(\Delta\approx 0\)). Length-stratified matrices are not, because support changes. Purpose–time is expected to split: one cell near \(\mathbf{R}\), one cell with \(\Delta>0\), one cell near neither \(\mathbf{R}\) nor \(\mathbf{R}^{\top}\).

### Observed-overlap tables

Table 2: purpose and time of day, including both CPC columns and \(\Delta\). Table 3: day, sex, age. Table 4: trip-length bands. Table 5: additional published attributes. Figures 1–2: CPC against \(\lambda\mathbf{R}\) versus CPC against \(\lambda\mathbf{R}^{\top}\), and CPC by partition type. Figure 4: CPC versus \(N\) and versus intra-zonal share.

## Experimental design

Condition S scores two observed matrices. Condition G scores a generated matrix against an observed target. Zero-shot generators see only \(\mathbf{R}\) during fitting. Few-shot neuroGravity additionally sees a declared subset of target edges, never the held-out target cells that are scored.

### Condition S pipeline

For each study area and each pre-specified partition:

1. Build \(\mathbf{T}\) by summing `n_trips` (or the Lombardy motive–mode cell) over every attribute not used to define the partition. Missing demographic tags in MITMA are kept in the unsliced total and dropped only from the tagged partition.
2. Build \(\mathbf{R}\) as above, on the same zones and the same date window.
3. If \(\sum R_{ij}=0\), skip the area.
4. Set \(\lambda=(\sum T_{ij})/(\sum R_{ij})\) and \(\mathbf{R}^{\downarrow}=\lambda\mathbf{R}\). Score \((\mathbf{T},\mathbf{R}^{\downarrow})\) and \((\mathbf{T},\lambda\mathbf{R}^{\top})\).
5. Score the independence table \(O_i^{T}D_j^{T}/\sum_k D_k^{T}\) against \(\mathbf{T}\) as a null.
6. Store \(\lambda\), \(h_{\mathrm{intra}}(\mathbf{T})\), CPC, \(\Delta\), and the companion / diagnostic columns required by the table contract.

Length-band matrices zero every cell whose vendor distance label lies outside the band, including intra-zonal cells if \(d_{ii}\) falls outside. They are therefore a change of support, not a reweighting of \(\mathbf{R}\).

Where \(D\ge 2\) valid weekdays exist, the same scores are recomputed on each day and summarised by the mean and a bootstrap percentile interval (\(1{,}000\) resamples of days). Split-half CPC of a partition against itself—random disjoint halves of days, mass-matched—is the noise ceiling for that partition. Lombardy has no day stack; only the published average weekday is scored.

### Condition G pipeline

Training data are always origin rows of \(\mathbf{R}\) in the training zone set. Target outflows \(O_i^{T}\) come from the observed partition being generated, including held-out origins. No cell of \(\mathbf{T}\) enters the loss or the deterrence calibration.

Production-constrained emission
\[
\widehat{T}_{ij}=O_i^{T}\,p_{j\mid i}.
\]
Row sums match by construction. Scores are on \((\widehat{\mathbf{T}},\mathbf{T})\) with no extra \(\lambda\).

IPF / Furness is a second emission rule, not a law. Seed is \(\mathbf{R}\) (or the gravity mean when a zero seed cell must be filled). Row totals \(O_i^{T}\). Two column-margin variants are run and labelled: (G-row) target inflows \(D_j^{T}\); (G-row-traincol) column totals predicted from training-origin inflows of \(\mathbf{R}\), scaled to \(\sum_i O_i^{T}\). Iteration stops at relative margin error \(10^{-4}\) or \(1{,}000\) cycles.

### Spatial split (mechanism subset)

The mechanism subset is the study areas where OSM features can be built and \(N\) is large enough for a five-block split (target: \(N\ge 80\)). Panel areas below that threshold receive Condition S and full-tessellation gravity / radiation only (Table 7).

Blocks are computed on the zone adjacency graph (queen contiguity of polygons). The graph is partitioned into five connected blocks of origins by greedy graph growing from five seeds placed at a farthest-point start, so that block sizes differ by at most one zone. Each fold trains on four blocks and generates the fifth. Destinations remain the full tessellation; only origins are held out. Table 6 reports the unweighted mean of the five held-out scores for the three primary purpose–time targets (other AM, home PM, home AM).

A random origin hold-out of the same size is stored in the supplement.

### Scale experiment

From the native tessellation, two coarser layers are built. (i) Official dissolve: MITMA districts to municipalities; Lombardy zones to comuni where a key exists. (ii) Random contiguous merge: repeatedly join a zone to a random neighbour until \(N\) is halved, repeated for three independent realisations. Condition S is rerun on every coarser matrix. Table 8 reports, across partitions, the Spearman correlation of CPC\((\mathbf{T},\lambda\mathbf{R})\) with the native-tessellation CPC, and the fraction of partitions that keep the sign of \(\Delta\).

### Attractiveness regression

On the mechanism subset only, OLS of destination inflows of \(\mathbf{R}\) on residential population and on OSM POI counts (all POIs; school POIs when the study partition is scored). That regression \(R^{2}\) is reported separately from the log-flow \(R^{2}\) in the GOF columns.

## Models

Laws output \(p_{j\mid i}\). Models output \(\widehat{\mathbf{T}}\).

### Gravity

\[
p_{j\mid i}=\frac{m_j f(d_{ij})}{\sum_k m_k f(d_{ik})},\qquad f(d)=d^{-\beta}\ \text{or}\ e^{-\beta d}.
\]
Default attractiveness \(m_j\) is residential population. A jobs attractor is a robustness swap where employment is published. \(\beta>0\) is calibrated by maximum likelihood on training-origin outflows of \(\mathbf{R}\):
\[
\ell(\beta)=\sum_{i\in\mathrm{train}}\sum_j R_{ij}\log p_{j\mid i}(\beta),
\]
with a bounded scalar search (power: \(\beta\in[0.1,3]\); exponential: \(\beta\) in inverse kilometres over \([0.01,2]\)). Intra-zonal cells enter the likelihood. One \(\beta\) per study area per deterrence, not a pooled national \(\beta\).

### Radiation

\[
p_{j\mid i}=\frac{m_i m_j}{(m_i+s_{ij})(m_i+m_j+s_{ij})},\qquad i\neq j,
\]
with \(p_{i\mid i}\) set to the observed intra-zonal share of origin \(i\) in \(\mathbf{R}\) on training origins, and to the area-wide intra-zonal share of \(\mathbf{R}\) on held-out origins. \(s_{ij}\) is the population (default) or POI-count mass in the closed Euclidean disk of radius \(d_{ij}\) centred at the centroid of \(i\), excluding \(i\) and \(j\). No free parameter.

### Deep Gravity

Follow Simini et al. without substituting a new feature list. Input for pair \((i,j)\): origin feature vector, destination feature vector, \(d_{ij}\), and the two populations (\(39\) scalars after the published OSM stack). Location features, each divided by polygon area except distance and raw population:

- land use, area in km\(^2\): residential, commercial, industrial, retail, natural;
- road length in km: residential, main, other;
- POI / building counts: transport, food, health, education, retail and the remaining published groups in the Deep Gravity OSM query.

Architecture: the published feed-forward stack with LeakyReLU hidden layers and a softmax over destinations of the same origin. Loss is cross-entropy on destination shares of \(\mathbf{R}\)
\[
H=-\sum_{i\in\mathrm{train}}\sum_j \frac{R_{ij}}{O_i^{R}}\log p_{j\mid i}.
\]
Training hyperparameters copied from the paper unless the origin count is smaller than a batch: \(20\) epochs, RMSprop with momentum \(0.9\), learning rate \(5\times 10^{-6}\), batch of \(64\) origins, at most \(512\) sampled destinations per origin when \(N>512\). Features are computed once per tessellation from a frozen OSM extract. The network is trained again in each spatial fold. It is never trained on \(\mathbf{T}\).

### Random forest control

Not a law. For each training origin, destinations are samples with weight \(R_{ij}\). Features: \(\log m_i\), \(\log m_j\), \(\log d_{ij}\), and the same area-normalised OSM counts used by Deep Gravity. The forest predicts a score that is softmax-normalised over \(j\) for each \(i\). Default: \(500\) trees, minimum leaf \(20\) origin–destination pairs. Used only on the mechanism subset.

### Closed-form gravity-like law

Cabanas-Tirapu et al. replace a hand-chosen deterrence with a gravity-like expression discovered by Bayesian machine scientist search on population and distance. On the mechanism subset the published closed form, or the form re-discovered on training origins of \(\mathbf{R}\), is run production-constrained as an extra gravity-family baseline.

### meta-Gravity and neuroGravity

Yang et al. factor reconstruction into a physics-informed prior and a graph residual. meta-Gravity predicts pair-specific parameters from concatenated origin and destination features \(h_i^{(0)}\oplus h_j^{(0)}\):
\[
\hat F_{ij}^{g}=\frac{G(h_i^{(0)}\oplus h_j^{(0)})\,P_i P_j}{d_{ij}^{\alpha(h_i^{(0)}\oplus h_j^{(0)})}}.
\]
Those estimates, with \(d_{ij}\), initialise edge features. An edge-enhanced graph transformer then updates node and edge embeddings; a final MLP emits \(\log\hat F_{ij}\). Training uses the published Huber reconstruction loss on both heads. Feature vectors follow the neuroGravity built-environment stack (population, land use, roads, POI) computed on the same OSM extract as Deep Gravity.

Three operating modes, scored with CPC and raw-flow \(R^{2}\) on the target partition:

1. **meta-Gravity alone**, production-constrained from \(\mathbf{R}\) (zero-shot prior).
2. **neuroGravity zero-shot.** The transformer is trained on \(\mathbf{R}\) in the spatial training blocks and applied to held-out origins with target outflows \(O_i^{T}\). No target-partition edge enters training.
3. **neuroGravity few-shot.** After the HBW pre-training in (2), a declared subset of target-partition edges is revealed. Default masks follow Yang et al.: random 1% of internal OD pairs; random 10% of internal pairs; and internal-only edges among a random 10% of zones. Observed edges are weighted by a softmax of the meta-Gravity prior at temperature \(\tau=2\). Held-out target cells are the test set. Few-shot never uses the Condition G spatial-block test origins as observed edges.

A LightGBM link predictor, when used, is trained on support in \(\mathbf{R}\) and applied unchanged to the target.

### IPF and oracle

IPF as in the Condition G pipeline. Oracle: refit the gravity power law (and, on the mechanism subset, Deep Gravity) on the target partition’s training origins, then generate held-out origins with that target-specific \(p_{j\mid i}\). The oracle measures how much of the gap is the wrong kernel rather than unpredictable destinations.

## Discussion

The comparison is whether morning HBW, its transpose, a zero-shot commuting generator, or a few-shot neuroGravity adaptation recovers each published cut of the same tessellation.

If day, sex and age partitions remain near \(\mathbf{R}\) under Condition S, a commuting kernel is already a usable prior for those cuts. If length-stratified matrices diverge on CPC and on support, a single deterrence fitted on all lengths does not replace a length-specific table. If evening home-bound flows overlap \(\mathbf{R}^{\top}\) more than \(\mathbf{R}\), the relevant prior is transposed commuting. If morning home-bound flows overlap neither, the destination field has to be estimated on that partition.

WorldMove and the Rong et al. global commuting product embed one HBW-like kernel in every downstream city. Deep Gravity supplies the geographic feature route to that same kernel. Vo et al. show that mode-by-hour cubes can be rebuilt and scored with CPC; few-shot neuroGravity asks how many target edges are required before a commuting-informed graph residual closes the remaining gap. Limits follow the products: inferred activity is not a travel diary; time bins differ across vendors; household surveys are typically one weekday; coarse tessellations inflate CPC by hiding destination change inside intra-zonal mass.

## Implementation notes

Supplementary Table S1 lists, for each study area: product version, date window and excluded outage dates, hour endpoints if they differ from the defaults above, CRS, population vintage, OSM extract date, and whether the tessellation is district-native or a municipal dissolve.

Matrices are stored as long tables with columns `origin`, `destination`, `flow`, `partition`, `date` (nullable), equivalent to a scikit-mobility `FlowDataFrame`. CPC uses the Lenormand / scikit-mobility denominator, not a “corrected” form. Gravity and radiation emission use the library’s production-constrained generators so that \(\sum_j\widehat{T}_{ij}=O_i^{T}\) is an assertion, not an informal rescale. Deep Gravity training follows the public scikit-mobility implementation with the hyperparameters above.

JSD uses \(20\) log-spaced bins on \([0,\max(A_{ij},B_{ij})]\) of the two matrices being compared, with empty bins kept so that the histogram support is shared. NRMSE uses RMSE over all \(N^{2}\) cells, including zeros, divided by the mean cell of the first argument. Pearson drops cells with \(A_{ij}+B_{ij}=0\). CPL treats a cell as present if the flow is strictly positive after aggregation; MITMA fractional `n_trips` below \(10^{-6}\) is treated as zero.

## Data availability

To be completed after reprocessing. Mobile-network OD: national open releases. Planning matrices and household surveys: official open data. Tessellations and codebooks: repository.

## Code availability

To be completed. Scripts for Condition S, Condition G, spatial blocks and tessellation aggregation.

## Acknowledgements

*[blank]*

## References

Luca, M., Barlacchi, G., Lepri, B. & Pappalardo, L. A survey on deep learning for human mobility. *ACM Comput. Surv.* **55**, 7 (2021).

Lenormand, M., Huet, S., Gargiulo, F. & Deffuant, G. A universal model of commuting networks. *PLoS ONE* **7**, e45985 (2012).

Lenormand, M., Bassolas, A. & Ramasco, J. J. Systematic comparison of trip distribution laws and models. *J. Transp. Geogr.* **51**, 158–169 (2016).

Simini, F., González, M. C., Maritan, A. & Barabási, A.-L. A universal model for mobility and migration patterns. *Nature* **484**, 96–100 (2012).

Simini, F., Barlacchi, G., Luca, M. & Pappalardo, L. A Deep Gravity model for mobility flows generation. *Nat. Commun.* **12**, 6576 (2021).

Cabanas-Tirapu, O. et al. Human mobility is well described by closed-form gravity-like models learned automatically from data. *Nat. Commun.* **16**, 1336 (2025).

Vo, K. D., Ham, S. W., Roy, M., Mishra, S. & Bansal, P. Uncovering latent urban mobility patterns via smart-card and survey data fusion. *Nat. Commun.* **17**, 7102 (2026).

Yang, J. et al. Transferable human mobility network reconstruction with neuroGravity. *Nat. Comput. Sci.* **6**, 630–641 (2026).

Rong, C., Ding, J., Li, M. & Li, Y. A global intra-city commuting origin-destination flow dataset for urban sustainable development. *Sci. Data* (2026).

Pappalardo, L. et al. scikit-mobility: a Python library for the analysis, generation and risk assessment of mobility data. *J. Stat. Softw.* (2022).

Schläpfer, M. et al. The universal visitation law of human mobility. *Nature* **593**, 522–527 (2021).

Yuan, Y. et al. DeepMobility: micro–macro collaborative generation of human mobility. *PNAS Nexus* (2025).

Yuan, Y. et al. WorldMove, a global open data for human mobility. *Sci. Data* **13**, 549 (2026).

Li, Z. et al. Sensing and modelling of urban human mobility. *Geomatics Inf. Sci. Wuhan Univ.* (2025).

Gaskin, T. & Abel, G. J. Deep learning of global human migration. *Nature* (2026).

Chi, G. & Abel, G. J. Measuring global monthly human migration flows. *Proc. Natl Acad. Sci. USA* (2025).

Ministerio de Transportes y Movilidad Sostenible. Estudio de movilidad de viajeros de ámbito nacional. Methodological report v8 (2024).

Regione Lombardia. Matrice OD2016 passeggeri.

## Tables

**Table 1.** Study areas.

```
| Study area | Product | \(N\) | Intra-zonal share | Days | Attributes present |
| ---------- | ------- | ----- | ----------------- | ---- | ------------------ |
|            |         |       |                   |      |                    |
```

**Table 2.** Condition S, purpose and time of day.

```
| Study area | Partition | \(\lambda\) | CPC(\(\mathbf{T},\lambda\mathbf{R}\)) | CPC(\(\mathbf{T},\lambda\mathbf{R}^{\top}\)) | \(\Delta\) | CPL | CPCd | \(R^{2}\) | NRMSE | JSD |
| ---------- | --------- | ----------- | ------------------------------------- | -------------------------------------------- | ---------- | --- | ---- | --------- | ----- | --- |
|            | other, AM |             |                                       |                                              |            |     |      |           |       |     |
|            | home, PM  |             |                                       |                                              |            |     |      |           |       |     |
|            | home, AM  |             |                                       |                                              |            |     |      |           |       |     |
```

**Table 3.** Condition S, day, sex, age.

```
| Study area | Partition | \(\lambda\) | CPC(\(\mathbf{T},\lambda\mathbf{R}\)) | \(\Delta\) | CPL | CPCd | \(R^{2}\) | NRMSE | JSD |
| ---------- | --------- | ----------- | ------------------------------------- | ---------- | --- | ---- | --------- | ----- | --- |
|            |           |             |                                       |            |     |      |           |       |     |
```

**Table 4.** Condition S, trip-length bands.

```
| Study area | Band | CPC | CPL | CPCd | \(R^{2}\) | NRMSE | JSD |
| ---------- | ---- | --- | --- | ---- | --------- | ----- | --- |
|            |      |     |     |      |           |       |     |
```

**Table 5.** Condition S, additional published attributes.

```
| Study area | Partition | CPC | \(\Delta\) | CPL | CPCd | \(R^{2}\) | NRMSE | JSD |
| ---------- | --------- | --- | ---------- | --- | ---- | --------- | ----- | --- |
|            |           |     |            |     |      |           |       |     |
```

**Table 6.** Condition G, mean over held-out origin blocks.

```
| Study area | Target | Law / model             | CPC | CPL | CPCd | \(R^{2}\) | NRMSE | JSD |
| ---------- | ------ | ----------------------- | --- | --- | ---- | --------- | ----- | --- |
|            |        | Gravity, power          |     |     |      |           |       |     |
|            |        | Gravity, exp.           |     |     |      |           |       |     |
|            |        | Radiation               |     |     |      |           |       |     |
|            |        | Deep Gravity            |     |     |      |           |       |     |
|            |        | Closed-form gravity     |     |     |      |           |       |     |
|            |        | meta-Gravity            |     |     |      |           |       |     |
|            |        | neuroGravity, zero-shot |     |     |      |           |       |     |
|            |        | neuroGravity, 1% edges  |     |     |      |           |       |     |
|            |        | neuroGravity, 10% edges |     |     |      |           |       |     |
|            |        | Random forest           |     |     |      |           |       |     |
|            |        | IPF from \(\mathbf{R}\) |     |     |      |           |       |     |
|            |        | Oracle                  |     |     |      |           |       |     |
```

**Table 7.** Production-constrained gravity, radiation, meta-Gravity and neuroGravity zero-shot on the panel, no origin hold-out. Schema as Table 6.

**Table 8.** Spatial scale. Spearman rank correlation of partition-wise CPC with the original tessellation; fraction of partitions that keep the sign of \(\Delta\).

## Figure legends

**Fig. 1** Condition S. Each point is one partition in one study area. Horizontal axis: CPC(\(\mathbf{T},\lambda\mathbf{R}\)). Vertical axis: CPC(\(\mathbf{T},\lambda\mathbf{R}^{\top}\)).

**Fig. 2** Condition S by partition type (purpose–time, day/sex/age, length band, other attributes).

**Fig. 3** Condition G. CPC of each generator, including meta-Gravity and few-shot neuroGravity, against the three purpose–time targets, mechanism subset.

**Fig. 4** CPC(\(\mathbf{T},\lambda\mathbf{R}\)) versus number of zones and versus intra-zonal share.
