/*
 * bundle_fwer_omp.cpp
 *
 * Exact C++/OpenMP implementation of the deterministic bundle stages used by
 * bundle_fwer.py.  It consumes sparse .bsp files from
 * permutationTest_cuda_bundle and keeps the established Python/COFFEE-DAC
 * routines untouched as a regression oracle.
 *
 * The important algorithmic change is strict bundle construction.  The
 * Python oracle examines every pair of edges incident on a shared voxel.  We
 * instead index each incident edge by its free endpoint and query only mask
 * voxels in the Chebyshev neighbourhood.  This produces the same union-find
 * components in expected O(E * neighbourhood_size), rather than O(sum d_v^2).
 */

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <boost/math/distributions/students_t.hpp>
#include <omp.h>


static constexpr uint32_t SPARSE_MAGIC = 0x4C444E42u; /* "BNDL" */
static constexpr uint32_t SPARSE_VERSION_FIXED = 1u;
static constexpr uint32_t SPARSE_VERSION_DF_AWARE = 2u;
static constexpr uint32_t SPARSE_VERSION_DF_STORED = 3u;
static constexpr uint32_t SPARSE_FLAG_DF_AWARE = 1u;
static constexpr uint32_t SPARSE_FLAG_DF_STORED = 2u;
static constexpr uint32_t NONE = std::numeric_limits<uint32_t>::max();
static constexpr uint32_t T_LOOKUP_STEPS_PER_DF = 4096u;


#pragma pack(push, 1)
struct SparseHeader {
    uint32_t magic;
    uint32_t version;
    uint64_t permutation;
    uint64_t n_records;
    uint64_t n_voxels;
    uint64_t n_total_edges;
    float threshold;
    uint32_t reserved;
};

struct SparseRecordV1 {
    uint64_t edge_index;
    float tstat;
};

struct SparseRecordV2 {
    uint64_t edge_index;
    float tstat;
    float excess;
};

struct SparseRecordV3 {
    uint64_t edge_index;
    float tstat;
    float degrees_of_freedom;
};
#pragma pack(pop)

static_assert(sizeof(SparseHeader) == 48, "unexpected sparse header size");
static_assert(sizeof(SparseRecordV1) == 12, "unexpected v1 sparse record size");
static_assert(sizeof(SparseRecordV2) == 16, "unexpected v2 sparse record size");
static_assert(sizeof(SparseRecordV3) == 16, "unexpected v3 sparse record size");


struct SparseRecord {
    uint64_t edge_index;
    float tstat;
    float auxiliary;
};


struct Coord {
    int32_t x;
    int32_t y;
    int32_t z;

    bool operator==(const Coord &other) const noexcept
    {
        return x == other.x && y == other.y && z == other.z;
    }
};


struct CoordHash {
    size_t operator()(const Coord &coord) const noexcept
    {
        uint64_t value = static_cast<uint32_t>(coord.x);
        value = value * 0x9E3779B185EBCA87ULL
            ^ static_cast<uint32_t>(coord.y);
        value = value * 0xC2B2AE3D27D4EB4FULL
            ^ static_cast<uint32_t>(coord.z);
        value ^= value >> 33;
        return static_cast<size_t>(value);
    }
};


struct MaskGeometry {
    std::vector<Coord> coordinates;
    std::vector<std::vector<uint32_t>> neighbours;
    int neighbour_radius = 0;
};


struct Edge {
    uint32_t endpoint1;
    uint32_t endpoint2;
    float tstat;
    double excess;
    uint32_t label;
};


struct Incidence {
    std::vector<uint64_t> offsets;
    std::vector<uint32_t> edge_indices;
};


struct BundleRow {
    uint32_t bundle;
    int sign;
    uint64_t edge_count;
    double mass;
    double statistic;
};


struct PermutationResult {
    uint64_t permutation = 0;
    uint64_t threshold_edges = 0;
    uint64_t retained_edges = 0;
    uint64_t bundles = 0;
    double max_statistic = 0.0;
    std::vector<Edge> observed_edges;
    std::vector<BundleRow> observed_bundles;
};


class DisjointSet {
public:
    explicit DisjointSet(size_t size)
        : parent_(size), rank_(size, 0)
    {
        std::iota(parent_.begin(), parent_.end(), 0u);
    }

    uint32_t find(uint32_t value)
    {
        uint32_t root = value;
        while (parent_[root] != root)
            root = parent_[root];
        while (parent_[value] != value) {
            uint32_t next = parent_[value];
            parent_[value] = root;
            value = next;
        }
        return root;
    }

    void unite(uint32_t first, uint32_t second)
    {
        uint32_t root_first = find(first);
        uint32_t root_second = find(second);
        if (root_first == root_second)
            return;
        if (rank_[root_first] < rank_[root_second])
            std::swap(root_first, root_second);
        parent_[root_second] = root_first;
        if (rank_[root_first] == rank_[root_second])
            rank_[root_first]++;
    }

private:
    std::vector<uint32_t> parent_;
    std::vector<uint8_t> rank_;
};


static size_t parse_size(const char *text, const char *name)
{
    errno = 0;
    char *end = nullptr;
    unsigned long long value = std::strtoull(text, &end, 10);
    if (errno || end == text || *end != '\0')
        throw std::runtime_error(std::string("invalid ") + name + ": " + text);
    return static_cast<size_t>(value);
}


static double parse_double(const char *text, const char *name)
{
    errno = 0;
    char *end = nullptr;
    double value = std::strtod(text, &end);
    if (errno || end == text || *end != '\0' || !std::isfinite(value))
        throw std::runtime_error(std::string("invalid ") + name + ": " + text);
    return value;
}


static std::vector<double> critical_t_lookup(size_t n_subjects,
                                             double two_sided_p)
{
    if (n_subjects < 4)
        throw std::runtime_error("Welch testing requires at least four subjects");
    uint32_t maximum_df = static_cast<uint32_t>(n_subjects) - 2u;
    size_t count = static_cast<size_t>(maximum_df - 1u)
        * T_LOOKUP_STEPS_PER_DF + 1u;
    std::vector<double> lookup(count);
    for (size_t index = 0; index < count; ++index) {
        double df = 1.0 + static_cast<double>(index) / T_LOOKUP_STEPS_PER_DF;
        boost::math::students_t_distribution<double> distribution(df);
        lookup[index] = boost::math::quantile(
            boost::math::complement(distribution, two_sided_p / 2.0));
    }
    return lookup;
}


static double interpolated_critical_t(double degrees_of_freedom,
                                      const std::vector<double> &lookup)
{
    double position = (degrees_of_freedom - 1.0) * T_LOOKUP_STEPS_PER_DF;
    position = std::min(std::max(position, 0.0),
                        static_cast<double>(lookup.size() - 1));
    size_t lower = static_cast<size_t>(std::floor(position));
    size_t upper = std::min(lower + 1, lookup.size() - 1);
    double fraction = position - static_cast<double>(lower);
    return lookup[lower] + fraction * (lookup[upper] - lookup[lower]);
}


static std::string sparse_path(const std::string &prefix, uint64_t permutation)
{
    std::ostringstream stream;
    stream << prefix << "_perm" << std::setw(6) << std::setfill('0')
           << permutation << ".bsp";
    return stream.str();
}


static MaskGeometry load_mask(const std::string &path, double neighbour_dist)
{
    std::ifstream stream(path);
    if (!stream)
        throw std::runtime_error("cannot open mask: " + path);

    MaskGeometry mask;
    std::string line;
    while (std::getline(stream, line)) {
        if (line.empty())
            continue;
        std::istringstream row(line);
        double x, y, z;
        if (!(row >> x >> y >> z))
            throw std::runtime_error("invalid mask row in " + path);
        mask.coordinates.push_back({
            static_cast<int32_t>(std::llround(x)),
            static_cast<int32_t>(std::llround(y)),
            static_cast<int32_t>(std::llround(z)),
        });
    }
    if (mask.coordinates.size() < 2)
        throw std::runtime_error("mask contains fewer than two voxels");

    std::unordered_map<Coord, uint32_t, CoordHash> coordinate_index;
    coordinate_index.reserve(mask.coordinates.size() * 2);
    for (uint32_t index = 0; index < mask.coordinates.size(); ++index) {
        if (!coordinate_index.emplace(mask.coordinates[index], index).second)
            throw std::runtime_error("mask contains duplicate coordinates");
    }

    int distance = static_cast<int>(std::ceil(neighbour_dist));
    mask.neighbour_radius = distance;
    mask.neighbours.resize(mask.coordinates.size());
    #pragma omp parallel for schedule(static)
    for (int64_t index = 0;
         index < static_cast<int64_t>(mask.coordinates.size()); ++index) {
        const Coord &coord = mask.coordinates[static_cast<size_t>(index)];
        std::vector<uint32_t> local;
        local.reserve(static_cast<size_t>((2 * distance + 1)
            * (2 * distance + 1) * (2 * distance + 1)));
        for (int dx = -distance; dx <= distance; ++dx) {
            for (int dy = -distance; dy <= distance; ++dy) {
                for (int dz = -distance; dz <= distance; ++dz) {
                    auto found = coordinate_index.find({
                        coord.x + dx, coord.y + dy, coord.z + dz,
                    });
                    if (found != coordinate_index.end())
                        local.push_back(found->second);
                }
            }
        }
        mask.neighbours[static_cast<size_t>(index)] = std::move(local);
    }
    return mask;
}


static std::pair<uint32_t, uint32_t> condensed_pair(
    uint64_t flat_index, uint64_t n_voxels)
{
    uint64_t n_edges = n_voxels * (n_voxels - 1) / 2;
    if (flat_index >= n_edges)
        throw std::runtime_error("sparse edge index lies outside upper triangle");

    long double a = static_cast<long double>(2 * n_voxels - 1);
    long double discriminant = a * a
        - static_cast<long double>(8) * flat_index;
    uint64_t row = static_cast<uint64_t>(
        std::floor((a - std::sqrt(discriminant)) / 2));
    auto row_start = [n_voxels](uint64_t value) {
        return value * (2 * n_voxels - value - 1) / 2;
    };
    while (row > 0 && row_start(row) > flat_index)
        --row;
    while (row + 1 < n_voxels && row_start(row + 1) <= flat_index)
        ++row;
    uint64_t column = row + 1 + flat_index - row_start(row);
    return {static_cast<uint32_t>(row), static_cast<uint32_t>(column)};
}


static std::pair<SparseHeader, std::vector<SparseRecord>> read_sparse(
    const std::string &path)
{
    std::ifstream stream(path, std::ios::binary);
    if (!stream)
        throw std::runtime_error("cannot open sparse input: " + path);

    SparseHeader header{};
    stream.read(reinterpret_cast<char *>(&header), sizeof(header));
    if (!stream || header.magic != SPARSE_MAGIC
            || (header.version != SPARSE_VERSION_FIXED
                && header.version != SPARSE_VERSION_DF_AWARE
                && header.version != SPARSE_VERSION_DF_STORED))
        throw std::runtime_error("invalid sparse header: " + path);
    if ((header.version != SPARSE_VERSION_FIXED)
            != ((header.reserved & SPARSE_FLAG_DF_AWARE) != 0))
        throw std::runtime_error("inconsistent sparse threshold mode: " + path);
    if ((header.version == SPARSE_VERSION_DF_STORED)
            != ((header.reserved & SPARSE_FLAG_DF_STORED) != 0))
        throw std::runtime_error("inconsistent sparse df-storage mode: " + path);
    if (header.n_records > static_cast<uint64_t>(
            std::numeric_limits<size_t>::max() / sizeof(SparseRecordV2)))
        throw std::runtime_error("sparse record count is too large");

    std::vector<SparseRecord> records(static_cast<size_t>(header.n_records));
    if (!records.empty()) {
        if (header.version == SPARSE_VERSION_FIXED) {
            std::vector<SparseRecordV1> raw(records.size());
            stream.read(reinterpret_cast<char *>(raw.data()),
                        static_cast<std::streamsize>(raw.size()
                            * sizeof(SparseRecordV1)));
            if (!stream)
                throw std::runtime_error("truncated v1 sparse records: " + path);
            for (size_t index = 0; index < records.size(); ++index) {
                records[index] = {
                    raw[index].edge_index,
                    raw[index].tstat,
                    0.f,
                };
            }
        } else if (header.version == SPARSE_VERSION_DF_AWARE) {
            std::vector<SparseRecordV2> raw(records.size());
            stream.read(reinterpret_cast<char *>(raw.data()),
                        static_cast<std::streamsize>(raw.size()
                            * sizeof(SparseRecordV2)));
            if (!stream)
                throw std::runtime_error("truncated v2 sparse records: " + path);
            for (size_t index = 0; index < records.size(); ++index)
                records[index] = {
                    raw[index].edge_index, raw[index].tstat, raw[index].excess,
                };
        } else {
            std::vector<SparseRecordV3> raw(records.size());
            stream.read(reinterpret_cast<char *>(raw.data()),
                        static_cast<std::streamsize>(raw.size()
                            * sizeof(SparseRecordV3)));
            if (!stream)
                throw std::runtime_error("truncated v3 sparse records: " + path);
            for (size_t index = 0; index < records.size(); ++index)
                records[index] = {
                    raw[index].edge_index, raw[index].tstat,
                    raw[index].degrees_of_freedom,
                };
        }
    }
    char trailing;
    if (stream.read(&trailing, 1))
        throw std::runtime_error("unexpected trailing sparse data: " + path);

    std::sort(records.begin(), records.end(),
              [](const SparseRecord &first, const SparseRecord &second) {
                  return first.edge_index < second.edge_index;
              });
    for (size_t index = 0; index < records.size(); ++index) {
        if (records[index].edge_index >= header.n_total_edges)
            throw std::runtime_error("out-of-range sparse edge index: " + path);
        if (!std::isfinite(records[index].tstat))
            throw std::runtime_error("non-finite sparse t-statistic: " + path);
        if (!std::isfinite(records[index].auxiliary)
                || (header.version == SPARSE_VERSION_DF_AWARE
                    && records[index].auxiliary < -1e-5f)
                || (header.version == SPARSE_VERSION_DF_STORED
                    && records[index].auxiliary < 1.f))
            throw std::runtime_error("invalid sparse auxiliary value: " + path);
        if (index && records[index - 1].edge_index
                == records[index].edge_index)
            throw std::runtime_error("duplicate sparse edge index: " + path);
    }
    return {header, std::move(records)};
}


static Incidence build_incidence(const std::vector<Edge> &edges,
                                 size_t n_voxels)
{
    Incidence incidence;
    incidence.offsets.assign(n_voxels + 1, 0);
    for (const Edge &edge : edges) {
        incidence.offsets[edge.endpoint1 + 1]++;
        incidence.offsets[edge.endpoint2 + 1]++;
    }
    std::partial_sum(incidence.offsets.begin(), incidence.offsets.end(),
                     incidence.offsets.begin());
    incidence.edge_indices.resize(edges.size() * 2);
    std::vector<uint64_t> cursor = incidence.offsets;
    for (uint32_t index = 0; index < edges.size(); ++index) {
        incidence.edge_indices[cursor[edges[index].endpoint1]++] = index;
        incidence.edge_indices[cursor[edges[index].endpoint2]++] = index;
    }
    return incidence;
}


static bool endpoint_has_other_edge(
    uint32_t voxel,
    uint32_t self,
    const Incidence &incidence,
    const std::vector<std::vector<uint32_t>> &neighbours,
    const std::vector<uint32_t> *labels = nullptr,
    uint32_t required_label = 0,
    const std::vector<uint8_t> *active = nullptr)
{
    for (uint32_t neighbour : neighbours[voxel]) {
        for (uint64_t position = incidence.offsets[neighbour];
             position < incidence.offsets[neighbour + 1]; ++position) {
            uint32_t other = incidence.edge_indices[position];
            if (other == self)
                continue;
            if (labels && (*labels)[other] != required_label)
                continue;
            if (active && !(*active)[other])
                continue;
            return true;
        }
    }
    return false;
}


static std::vector<Edge> filter_isolated(
    const std::vector<Edge> &edges,
    const MaskGeometry &mask)
{
    if (edges.empty())
        return {};
    Incidence incidence = build_incidence(edges, mask.coordinates.size());
    std::vector<uint8_t> keep(edges.size(), 0);
    #pragma omp parallel for schedule(static)
    for (int64_t index = 0; index < static_cast<int64_t>(edges.size()); ++index) {
        const Edge &edge = edges[static_cast<size_t>(index)];
        keep[static_cast<size_t>(index)] =
            endpoint_has_other_edge(edge.endpoint1,
                                    static_cast<uint32_t>(index), incidence,
                                    mask.neighbours)
            || endpoint_has_other_edge(edge.endpoint2,
                                       static_cast<uint32_t>(index), incidence,
                                       mask.neighbours);
    }
    std::vector<Edge> filtered;
    filtered.reserve(edges.size());
    for (size_t index = 0; index < edges.size(); ++index)
        if (keep[index])
            filtered.push_back(edges[index]);
    return filtered;
}


static uint32_t assign_strict_labels(std::vector<Edge> &edges,
                                     const MaskGeometry &mask,
                                     const Incidence &incidence)
{
    if (edges.empty())
        return 0;
    DisjointSet sets(edges.size());
    std::vector<uint32_t> stamp(mask.coordinates.size(), NONE);
    std::vector<uint32_t> edge_at_free(mask.coordinates.size(), NONE);

    for (uint32_t shared = 0; shared < mask.coordinates.size(); ++shared) {
        uint64_t begin = incidence.offsets[shared];
        uint64_t end = incidence.offsets[shared + 1];
        if (end - begin < 2)
            continue;

        for (uint64_t position = begin; position < end; ++position) {
            uint32_t edge_index = incidence.edge_indices[position];
            const Edge &edge = edges[edge_index];
            uint32_t free_endpoint = edge.endpoint1 == shared
                ? edge.endpoint2 : edge.endpoint1;
            stamp[free_endpoint] = shared;
            edge_at_free[free_endpoint] = edge_index;
        }
        for (uint64_t position = begin; position < end; ++position) {
            uint32_t edge_index = incidence.edge_indices[position];
            const Edge &edge = edges[edge_index];
            uint32_t free_endpoint = edge.endpoint1 == shared
                ? edge.endpoint2 : edge.endpoint1;
            for (uint32_t neighbour : mask.neighbours[free_endpoint]) {
                if (stamp[neighbour] == shared) {
                    uint32_t other = edge_at_free[neighbour];
                    if (other > edge_index)
                        sets.unite(edge_index, other);
                }
            }
        }
    }

    std::vector<uint32_t> root_label(edges.size(), NONE);
    uint32_t n_labels = 0;
    for (uint32_t index = 0; index < edges.size(); ++index) {
        uint32_t root = sets.find(index);
        if (root_label[root] == NONE)
            root_label[root] = n_labels++;
        edges[index].label = root_label[root];
    }
    return n_labels;
}


static bool within_radius(uint32_t first, uint32_t second,
                          const MaskGeometry &mask)
{
    const Coord &a = mask.coordinates[first];
    const Coord &b = mask.coordinates[second];
    int radius = mask.neighbour_radius;
    return std::abs(a.x - b.x) <= radius
        && std::abs(a.y - b.y) <= radius
        && std::abs(a.z - b.z) <= radius;
}


/*
 * Non-chaining, orientation-invariant endpoint-patch bundling.
 *
 * The strongest unassigned edge is a representative connection. Every edge
 * assigned to it must be orientable so that one endpoint lies within the
 * configured radius of representative endpoint A and its other endpoint lies
 * within that radius of representative endpoint B. Thus each endpoint patch
 * has a hard Chebyshev diameter <= 2 * radius; no transitive path can expand a
 * bundle beyond that envelope. Ties are broken by the stable sparse edge
 * order, which is the condensed edge index order.
 */
static uint32_t assign_bounded_labels(std::vector<Edge> &edges,
                                      const MaskGeometry &mask,
                                      const Incidence &incidence)
{
    if (edges.empty())
        return 0;
    std::vector<uint32_t> priority(edges.size());
    std::iota(priority.begin(), priority.end(), 0u);
    std::stable_sort(priority.begin(), priority.end(),
        [&edges](uint32_t first, uint32_t second) {
            if (edges[first].excess != edges[second].excess)
                return edges[first].excess > edges[second].excess;
            return first < second;
        });

    std::vector<uint32_t> labels(edges.size(), NONE);
    std::vector<uint32_t> seen(edges.size(), NONE);
    uint32_t n_labels = 0;
    for (uint32_t seed_index : priority) {
        if (labels[seed_index] != NONE)
            continue;
        const Edge &seed = edges[seed_index];
        uint32_t stamp = n_labels;
        labels[seed_index] = n_labels;

        for (uint32_t nearby : mask.neighbours[seed.endpoint1]) {
            for (uint64_t position = incidence.offsets[nearby];
                 position < incidence.offsets[nearby + 1]; ++position) {
                uint32_t candidate_index = incidence.edge_indices[position];
                if (labels[candidate_index] != NONE
                        || seen[candidate_index] == stamp)
                    continue;
                seen[candidate_index] = stamp;
                const Edge &candidate = edges[candidate_index];
                bool direct = within_radius(candidate.endpoint1,
                                            seed.endpoint1, mask)
                    && within_radius(candidate.endpoint2,
                                     seed.endpoint2, mask);
                bool swapped = within_radius(candidate.endpoint2,
                                             seed.endpoint1, mask)
                    && within_radius(candidate.endpoint1,
                                     seed.endpoint2, mask);
                if (direct || swapped)
                    labels[candidate_index] = n_labels;
            }
        }
        ++n_labels;
    }
    for (uint32_t index = 0; index < edges.size(); ++index)
        edges[index].label = labels[index];
    return n_labels;
}


static uint32_t relabel_by_descending_size(std::vector<Edge> &edges)
{
    if (edges.empty())
        return 0;
    uint32_t max_label = 0;
    for (const Edge &edge : edges)
        max_label = std::max(max_label, edge.label);
    std::vector<uint64_t> counts(static_cast<size_t>(max_label) + 1, 0);
    for (const Edge &edge : edges)
        counts[edge.label]++;

    std::vector<uint32_t> labels;
    for (uint32_t label = 0; label < counts.size(); ++label)
        if (counts[label])
            labels.push_back(label);
    std::stable_sort(labels.begin(), labels.end(),
        [&counts](uint32_t first, uint32_t second) {
            return counts[first] > counts[second];
        });
    std::vector<uint32_t> remap(counts.size(), NONE);
    for (uint32_t index = 0; index < labels.size(); ++index)
        remap[labels[index]] = index;
    for (Edge &edge : edges)
        edge.label = remap[edge.label];
    return static_cast<uint32_t>(labels.size());
}


static std::vector<Edge> select_active(const std::vector<Edge> &edges,
                                       const std::vector<uint8_t> &active)
{
    std::vector<Edge> output;
    output.reserve(edges.size());
    for (size_t index = 0; index < edges.size(); ++index)
        if (active[index])
            output.push_back(edges[index]);
    relabel_by_descending_size(output);
    return output;
}


static std::vector<std::vector<uint32_t>> component_edge_indices(
    const std::vector<Edge> &edges)
{
    uint32_t count = 0;
    for (const Edge &edge : edges)
        count = std::max(count, edge.label + 1);
    std::vector<std::vector<uint32_t>> components(count);
    for (uint32_t index = 0; index < edges.size(); ++index)
        components[edges[index].label].push_back(index);
    return components;
}


static std::vector<Edge> prune_intra_network_isolated(
    const std::vector<Edge> &edges,
    const MaskGeometry &mask)
{
    if (edges.empty())
        return {};
    Incidence incidence = build_incidence(edges, mask.coordinates.size());
    std::vector<std::vector<uint32_t>> components =
        component_edge_indices(edges);
    std::vector<uint32_t> labels(edges.size());
    for (size_t index = 0; index < edges.size(); ++index)
        labels[index] = edges[index].label;
    std::vector<uint8_t> active(edges.size(), 1);

    #pragma omp parallel for schedule(dynamic, 1)
    for (int64_t component_index = 0;
         component_index < static_cast<int64_t>(components.size());
         ++component_index) {
        const std::vector<uint32_t> &component =
            components[static_cast<size_t>(component_index)];
        std::vector<uint32_t> dropping;
        while (true) {
            dropping.clear();
            for (uint32_t edge_index : component) {
                if (!active[edge_index])
                    continue;
                const Edge &edge = edges[edge_index];
                bool endpoint1_ok = endpoint_has_other_edge(
                    edge.endpoint1, edge_index, incidence, mask.neighbours,
                    &labels, static_cast<uint32_t>(component_index), &active);
                bool endpoint2_ok = endpoint_has_other_edge(
                    edge.endpoint2, edge_index, incidence, mask.neighbours,
                    &labels, static_cast<uint32_t>(component_index), &active);
                if (!endpoint1_ok || !endpoint2_ok)
                    dropping.push_back(edge_index);
            }
            if (dropping.empty())
                break;
            for (uint32_t edge_index : dropping)
                active[edge_index] = 0;
        }
    }
    return select_active(edges, active);
}


static std::vector<Edge> filter_small_networks(
    const std::vector<Edge> &edges, size_t minimum_size)
{
    if (edges.empty())
        return {};
    uint32_t max_label = 0;
    for (const Edge &edge : edges)
        max_label = std::max(max_label, edge.label);
    std::vector<uint64_t> counts(static_cast<size_t>(max_label) + 1, 0);
    for (const Edge &edge : edges)
        counts[edge.label]++;
    std::vector<uint8_t> active(edges.size(), 0);
    for (size_t index = 0; index < edges.size(); ++index)
        active[index] = counts[edges[index].label] >= minimum_size;
    return select_active(edges, active);
}


static std::vector<Edge> prune_small_endpoint_clusters(
    const std::vector<Edge> &edges,
    const MaskGeometry &mask,
    size_t minimum_cluster_voxels)
{
    if (edges.empty())
        return {};
    std::vector<std::vector<uint32_t>> components =
        component_edge_indices(edges);
    std::vector<uint8_t> active(edges.size(), 1);

    #pragma omp parallel for schedule(dynamic, 1)
    for (int64_t component_index = 0;
         component_index < static_cast<int64_t>(components.size());
         ++component_index) {
        const std::vector<uint32_t> &component =
            components[static_cast<size_t>(component_index)];
        std::vector<uint32_t> voxels;
        voxels.reserve(component.size() * 2);
        for (uint32_t edge_index : component) {
            voxels.push_back(edges[edge_index].endpoint1);
            voxels.push_back(edges[edge_index].endpoint2);
        }
        std::sort(voxels.begin(), voxels.end());
        voxels.erase(std::unique(voxels.begin(), voxels.end()), voxels.end());

        std::unordered_map<uint32_t, uint32_t> local_index;
        local_index.reserve(voxels.size() * 2);
        for (uint32_t index = 0; index < voxels.size(); ++index)
            local_index.emplace(voxels[index], index);
        DisjointSet voxel_sets(voxels.size());
        for (uint32_t index = 0; index < voxels.size(); ++index) {
            for (uint32_t neighbour : mask.neighbours[voxels[index]]) {
                auto found = local_index.find(neighbour);
                if (found != local_index.end() && found->second > index)
                    voxel_sets.unite(index, found->second);
            }
        }
        std::vector<uint32_t> root_size(voxels.size(), 0);
        for (uint32_t index = 0; index < voxels.size(); ++index)
            root_size[voxel_sets.find(index)]++;
        for (uint32_t edge_index : component) {
            uint32_t first = local_index[edges[edge_index].endpoint1];
            uint32_t second = local_index[edges[edge_index].endpoint2];
            if (root_size[voxel_sets.find(first)] < minimum_cluster_voxels
                    || root_size[voxel_sets.find(second)]
                        < minimum_cluster_voxels)
                active[edge_index] = 0;
        }
    }
    return select_active(edges, active);
}


static std::vector<Edge> process_sign(std::vector<Edge> edges,
                                      const MaskGeometry &mask,
                                      size_t minimum_size,
                                      size_t minimum_cluster_voxels,
                                      bool bounded_bundles)
{
    edges = filter_isolated(edges, mask);
    if (edges.empty())
        return {};
    Incidence incidence = build_incidence(edges, mask.coordinates.size());
    if (bounded_bundles)
        assign_bounded_labels(edges, mask, incidence);
    else
        assign_strict_labels(edges, mask, incidence);
    edges = prune_intra_network_isolated(edges, mask);
    edges = filter_small_networks(edges, minimum_size);
    edges = prune_small_endpoint_clusters(
        edges, mask, minimum_cluster_voxels);
    edges = filter_small_networks(edges, minimum_size);
    return edges;
}


static PermutationResult process_permutation(
    const std::string &path,
    const MaskGeometry &mask,
    const std::string &statistic,
    double cluster_forming_threshold,
    bool df_aware,
    bool records_contain_df,
    const std::vector<double> &critical_t_values,
    size_t minimum_size,
    size_t minimum_cluster_voxels,
    bool bounded_bundles,
    bool retain_observed)
{
    auto sparse = read_sparse(path);
    const SparseHeader &header = sparse.first;
    std::vector<SparseRecord> &records = sparse.second;
    if (header.n_voxels != mask.coordinates.size())
        throw std::runtime_error("mask/sparse voxel count mismatch: " + path);
    if (header.n_total_edges != header.n_voxels * (header.n_voxels - 1) / 2)
        throw std::runtime_error("invalid sparse total-edge count: " + path);
    if ((header.version != SPARSE_VERSION_FIXED) != df_aware)
        throw std::runtime_error("sparse/requested threshold mode mismatch: " + path);
    if ((header.version == SPARSE_VERSION_DF_STORED) != records_contain_df)
        throw std::runtime_error("sparse/requested df-storage mode mismatch: " + path);
    if (!records_contain_df && std::fabs(static_cast<double>(header.threshold)
            - cluster_forming_threshold)
            > 1e-6 * std::max(1.0, std::fabs(cluster_forming_threshold)))
        throw std::runtime_error("sparse/requested threshold mismatch: " + path);

    std::vector<Edge> positive;
    std::vector<Edge> negative;
    positive.reserve(records.size() / 2);
    negative.reserve(records.size() / 2);
    for (const SparseRecord &record : records) {
        auto endpoints = condensed_pair(record.edge_index, header.n_voxels);
        double edge_critical_t = 0.0;
        if (records_contain_df) {
            edge_critical_t = interpolated_critical_t(
                static_cast<double>(record.auxiliary), critical_t_values);
        }
        double excess = records_contain_df
            ? std::fabs(static_cast<double>(record.tstat)) - edge_critical_t
            : (df_aware
                ? static_cast<double>(record.auxiliary)
            : std::fabs(static_cast<double>(record.tstat))
                - cluster_forming_threshold);
        if (excess < 0.0)
            continue;
        Edge edge{
            endpoints.first, endpoints.second, record.tstat, excess, 0,
        };
        if (record.tstat > 0)
            positive.push_back(edge);
        else if (record.tstat < 0)
            negative.push_back(edge);
    }
    records.clear();
    records.shrink_to_fit();
    uint64_t threshold_edge_count = positive.size() + negative.size();

    positive = process_sign(std::move(positive), mask, minimum_size,
                            minimum_cluster_voxels, bounded_bundles);
    negative = process_sign(std::move(negative), mask, minimum_size,
                            minimum_cluster_voxels, bounded_bundles);

    uint32_t positive_bundles = 0;
    for (const Edge &edge : positive)
        positive_bundles = std::max(positive_bundles, edge.label + 1);
    for (Edge &edge : negative)
        edge.label += positive_bundles;

    std::vector<Edge> combined;
    combined.reserve(positive.size() + negative.size());
    combined.insert(combined.end(), positive.begin(), positive.end());
    combined.insert(combined.end(), negative.begin(), negative.end());

    uint32_t n_bundles = 0;
    for (const Edge &edge : combined)
        n_bundles = std::max(n_bundles, edge.label + 1);
    std::vector<uint64_t> counts(n_bundles, 0);
    std::vector<double> masses(n_bundles, 0.0);
    std::vector<int> signs(n_bundles, 0);
    for (const Edge &edge : combined) {
        counts[edge.label]++;
        masses[edge.label] += static_cast<double>(edge.excess);
        if (signs[edge.label] == 0)
            signs[edge.label] = edge.tstat > 0 ? 1 : -1;
    }

    PermutationResult result;
    result.permutation = header.permutation;
    result.threshold_edges = threshold_edge_count;
    result.retained_edges = combined.size();
    result.bundles = n_bundles;
    for (uint32_t label = 0; label < n_bundles; ++label) {
        double value = statistic == "mass"
            ? masses[label] : static_cast<double>(counts[label]);
        result.max_statistic = std::max(result.max_statistic, value);
        if (retain_observed) {
            result.observed_bundles.push_back({
                label, signs[label], counts[label], masses[label], value,
            });
        }
    }
    if (retain_observed)
        result.observed_edges = std::move(combined);
    return result;
}


static void write_maxima(const std::string &path,
                         const std::vector<PermutationResult> &results)
{
    std::ofstream stream(path);
    if (!stream)
        throw std::runtime_error("cannot write maxima CSV: " + path);
    stream << "permutation,observed,threshold_edges,retained_edges,bundles,max_statistic\n";
    stream << std::setprecision(17);
    for (const PermutationResult &result : results) {
        stream << result.permutation << ','
               << (result.permutation == 0 ? "True" : "False") << ','
               << result.threshold_edges << ',' << result.retained_edges << ','
               << result.bundles << ',' << result.max_statistic << '\n';
    }
    if (!stream)
        throw std::runtime_error("failed writing maxima CSV: " + path);
}


static void write_observed_edges(const std::string &path,
                                 const PermutationResult &observed,
                                 const MaskGeometry &mask)
{
    std::ofstream stream(path);
    if (!stream)
        throw std::runtime_error("cannot write observed edges: " + path);
    stream << "i1,j1,k1,i2,j2,k2,pvalue,tstat,bundle,network\n";
    stream << std::setprecision(9);
    for (const Edge &edge : observed.observed_edges) {
        const Coord &first = mask.coordinates[edge.endpoint1];
        const Coord &second = mask.coordinates[edge.endpoint2];
        stream << first.x << ',' << first.y << ',' << first.z << ','
               << second.x << ',' << second.y << ',' << second.z
               << ",nan," << edge.tstat << ',' << edge.label << ','
               << edge.label << '\n';
    }
    if (!stream)
        throw std::runtime_error("failed writing observed edges: " + path);
}


static void write_observed_bundles(const std::string &path,
                                   const PermutationResult &observed)
{
    std::ofstream stream(path);
    if (!stream)
        throw std::runtime_error("cannot write observed bundles: " + path);
    stream << "bundle,sign,edge_count,mass,statistic\n";
    stream << std::setprecision(17);
    for (const BundleRow &bundle : observed.observed_bundles) {
        stream << bundle.bundle << ',' << bundle.sign << ','
               << bundle.edge_count << ',' << bundle.mass << ','
               << bundle.statistic << '\n';
    }
    if (!stream)
        throw std::runtime_error("failed writing observed bundles: " + path);
}


static void usage(const char *program)
{
    std::cerr
        << "Usage: " << program
        << " <mask.dump> <sparse_prefix> <start> <count> <mass|extent>"
        << " <threshold> <neighbor_dist> <min_size> <min_cluster_voxels>"
        << " <maxima.csv>"
        << " [--threads N] [--observed-edges FILE]"
        << " [--observed-bundles FILE] [--df-aware]"
        << " [--records-contain-df --subjects N] [--bounded-bundles]"
        << " [--strict-bundles] [--delete-inputs]\n";
}


int main(int argc, char **argv)
{
    try {
        if (argc < 11) {
            usage(argv[0]);
            return EXIT_FAILURE;
        }
        std::string mask_path = argv[1];
        std::string sparse_prefix = argv[2];
        size_t start = parse_size(argv[3], "start");
        size_t count = parse_size(argv[4], "count");
        std::string statistic = argv[5];
        double cluster_forming_threshold = parse_double(argv[6], "threshold");
        double neighbour_dist = parse_double(argv[7], "neighbor_dist");
        size_t minimum_size = parse_size(argv[8], "min_size");
        size_t minimum_cluster_voxels = parse_size(
            argv[9], "min_cluster_voxels");
        std::string maxima_path = argv[10];
        int threads = std::min(4, omp_get_max_threads());
        std::string observed_edges_path;
        std::string observed_bundles_path;
        bool delete_inputs = false;
        bool df_aware = false;
        bool records_contain_df = false;
        size_t n_subjects = 0;
#ifdef DEFAULT_BOUNDED_BUNDLES
        bool bounded_bundles = true;
#else
        bool bounded_bundles = false;
#endif

        for (int argument = 11; argument < argc; ++argument) {
            if (std::strcmp(argv[argument], "--threads") == 0
                    && argument + 1 < argc) {
                threads = static_cast<int>(parse_size(
                    argv[++argument], "threads"));
            } else if (std::strcmp(argv[argument], "--observed-edges") == 0
                    && argument + 1 < argc) {
                observed_edges_path = argv[++argument];
            } else if (std::strcmp(argv[argument], "--observed-bundles") == 0
                    && argument + 1 < argc) {
                observed_bundles_path = argv[++argument];
            } else if (std::strcmp(argv[argument], "--delete-inputs") == 0) {
                delete_inputs = true;
            } else if (std::strcmp(argv[argument], "--df-aware") == 0) {
                df_aware = true;
            } else if (std::strcmp(argv[argument], "--records-contain-df") == 0) {
                records_contain_df = true;
            } else if (std::strcmp(argv[argument], "--subjects") == 0
                    && argument + 1 < argc) {
                n_subjects = parse_size(argv[++argument], "subjects");
            } else if (std::strcmp(argv[argument], "--bounded-bundles") == 0) {
                bounded_bundles = true;
            } else if (std::strcmp(argv[argument], "--strict-bundles") == 0) {
                bounded_bundles = false;
            } else {
                throw std::runtime_error(
                    std::string("unknown or incomplete option: ")
                    + argv[argument]);
            }
        }
        if (count == 0 || minimum_size == 0
                || minimum_cluster_voxels == 0 || threads < 1)
            throw std::runtime_error("count, sizes, and threads must be positive");
        if (statistic != "mass" && statistic != "extent")
            throw std::runtime_error("statistic must be mass or extent");
        if (neighbour_dist < 0)
            throw std::runtime_error("neighbor_dist must be non-negative");
        if (cluster_forming_threshold < 0)
            throw std::runtime_error("threshold must be non-negative");
        if (records_contain_df && !df_aware)
            throw std::runtime_error("--records-contain-df requires --df-aware");
        if (records_contain_df && n_subjects < 4)
            throw std::runtime_error(
                "--records-contain-df requires --subjects >= 4");

        omp_set_dynamic(0);
        omp_set_num_threads(threads);
        MaskGeometry mask = load_mask(mask_path, neighbour_dist);
        std::vector<double> critical_t_values;
        if (records_contain_df)
            critical_t_values = critical_t_lookup(
                n_subjects, cluster_forming_threshold);
        std::vector<PermutationResult> results(count);
        std::vector<std::string> paths(count);
        for (size_t local = 0; local < count; ++local)
            paths[local] = sparse_path(sparse_prefix, start + local);

        std::atomic<bool> failed(false);
        std::string failure_message;
        #pragma omp parallel for schedule(dynamic, 1)
        for (int64_t local = 0; local < static_cast<int64_t>(count); ++local) {
            if (failed.load())
                continue;
            try {
                uint64_t permutation = start + static_cast<size_t>(local);
                bool observed = permutation == 0;
                results[static_cast<size_t>(local)] = process_permutation(
                    paths[static_cast<size_t>(local)], mask, statistic,
                    cluster_forming_threshold,
                    df_aware, records_contain_df,
                    critical_t_values,
                    minimum_size, minimum_cluster_voxels,
                    bounded_bundles, observed);
                if (results[static_cast<size_t>(local)].permutation
                        != permutation)
                    throw std::runtime_error("sparse permutation index mismatch");
                #pragma omp critical(bundle_progress)
                std::cout << "[bundle " << permutation << "] threshold="
                          << results[static_cast<size_t>(local)].threshold_edges
                          << " retained="
                          << results[static_cast<size_t>(local)].retained_edges
                          << " bundles="
                          << results[static_cast<size_t>(local)].bundles
                          << " max="
                          << results[static_cast<size_t>(local)].max_statistic
                          << std::endl;
            } catch (const std::exception &error) {
                failed.store(true);
                #pragma omp critical(bundle_failure)
                if (failure_message.empty())
                    failure_message = error.what();
            }
        }
        if (failed.load())
            throw std::runtime_error(failure_message);

        write_maxima(maxima_path, results);
        if (start == 0) {
            if (!observed_edges_path.empty())
                write_observed_edges(observed_edges_path, results[0], mask);
            if (!observed_bundles_path.empty())
                write_observed_bundles(observed_bundles_path, results[0]);
        }
        if (delete_inputs) {
            for (const std::string &path : paths) {
                if (std::remove(path.c_str()) != 0)
                    throw std::runtime_error("cannot delete sparse input: " + path);
            }
        }
        return EXIT_SUCCESS;
    } catch (const std::exception &error) {
        std::cerr << "ERROR: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
