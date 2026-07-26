#pragma once
// Collision mesh load + grid
// Extracted from RocketSimCuda.cu (modular split; same TU via include).

// ============================================================================
// Collision mesh loading (CMF format, vertices already in BT units)
// ============================================================================

static bool loadCollisionMeshes(
    const char* collisionMeshesPath,
    std::vector<MeshTriangle>& outTris
) {
    namespace fs = std::filesystem;

    fs::path base = fs::path(collisionMeshesPath) / "soccar";
    if (!fs::exists(base)) {
        fprintf(stderr, "RocketSimCuda: collision mesh folder not found: %s\n", base.string().c_str());
        return false;
    }

    int meshId = 0;
    for (const auto& entry : fs::directory_iterator(base)) {
        if (!entry.is_regular_file() || entry.path().extension() != ".cmf")
            continue;

        std::ifstream in(entry.path(), std::ios::binary);
        if (!in)
            continue;

        int32_t numTris = 0, numVerts = 0;
        in.read(reinterpret_cast<char*>(&numTris), 4);
        in.read(reinterpret_cast<char*>(&numVerts), 4);
        if (numTris <= 0 || numVerts <= 0 || numTris > 1000000 || numVerts > 1000000) {
            fprintf(stderr, "RocketSimCuda: bad CMF header in %s\n", entry.path().string().c_str());
            continue;
        }

        std::vector<int32_t> indices(numTris * 3);
        std::vector<float> verts(numVerts * 3);
        in.read(reinterpret_cast<char*>(indices.data()), numTris * 3 * 4);
        in.read(reinterpret_cast<char*>(verts.data()), numVerts * 3 * 4);
        if (!in) {
            fprintf(stderr, "RocketSimCuda: truncated CMF: %s\n", entry.path().string().c_str());
            continue;
        }

        for (int t = 0; t < numTris; t++) {
            int i0 = indices[t * 3 + 0], i1 = indices[t * 3 + 1], i2 = indices[t * 3 + 2];
            if (i0 < 0 || i0 >= numVerts || i1 < 0 || i1 >= numVerts || i2 < 0 || i2 >= numVerts)
                continue;
            MeshTriangle tri;
            tri.v0 = {verts[i0 * 3], verts[i0 * 3 + 1], verts[i0 * 3 + 2]};
            tri.v1 = {verts[i1 * 3], verts[i1 * 3 + 1], verts[i1 * 3 + 2]};
            tri.v2 = {verts[i2 * 3], verts[i2 * 3 + 1], verts[i2 * 3 + 2]};
            tri.meshId = meshId;
            outTris.push_back(tri);
        }
        meshId++;
    }

    return !outTris.empty();
}

// Uniform grid over the arena (BT units).
static void buildMeshGrid(
    const std::vector<MeshTriangle>& tris,
    MeshGridConfig& cfg,
    std::vector<int>& cellStart,
    std::vector<int>& cellTris
) {
    constexpr float CELL = 4.f;  // BT (= 200 UU)

    Vec3 mn = {1e30f, 1e30f, 1e30f}, mx = {-1e30f, -1e30f, -1e30f};
    for (const auto& t : tris) {
        const Vec3* vs[3] = {&t.v0, &t.v1, &t.v2};
        for (auto* v : vs) {
            mn.x = fminf(mn.x, v->x); mn.y = fminf(mn.y, v->y); mn.z = fminf(mn.z, v->z);
            mx.x = fmaxf(mx.x, v->x); mx.y = fmaxf(mx.y, v->y); mx.z = fmaxf(mx.z, v->z);
        }
    }
    // pad so queries at the boundary clamp inside
    mn.x -= 1.f; mn.y -= 1.f; mn.z -= 1.f;
    mx.x += 1.f; mx.y += 1.f; mx.z += 1.f;

    cfg.minPos = mn;
    cfg.nx = (int)ceilf((mx.x - mn.x) / CELL);
    cfg.ny = (int)ceilf((mx.y - mn.y) / CELL);
    cfg.nz = (int)ceilf((mx.z - mn.z) / CELL);
    cfg.invCellSize = {1.f / CELL, 1.f / CELL, 1.f / CELL};

    int numCells = cfg.nx * cfg.ny * cfg.nz;
    std::vector<std::vector<int>> cells(numCells);

    for (int i = 0; i < (int)tris.size(); i++) {
        const auto& t = tris[i];
        Vec3 tmn = {fminf(t.v0.x, fminf(t.v1.x, t.v2.x)),
                    fminf(t.v0.y, fminf(t.v1.y, t.v2.y)),
                    fminf(t.v0.z, fminf(t.v1.z, t.v2.z))};
        Vec3 tmx = {fmaxf(t.v0.x, fmaxf(t.v1.x, t.v2.x)),
                    fmaxf(t.v0.y, fmaxf(t.v1.y, t.v2.y)),
                    fmaxf(t.v0.z, fmaxf(t.v1.z, t.v2.z))};

        int cx0 = (int)floorf((tmn.x - mn.x) / CELL), cx1 = (int)floorf((tmx.x - mn.x) / CELL);
        int cy0 = (int)floorf((tmn.y - mn.y) / CELL), cy1 = (int)floorf((tmx.y - mn.y) / CELL);
        int cz0 = (int)floorf((tmn.z - mn.z) / CELL), cz1 = (int)floorf((tmx.z - mn.z) / CELL);
        cx0 = cx0 < 0 ? 0 : cx0; cy0 = cy0 < 0 ? 0 : cy0; cz0 = cz0 < 0 ? 0 : cz0;
        cx1 = cx1 >= cfg.nx ? cfg.nx - 1 : cx1;
        cy1 = cy1 >= cfg.ny ? cfg.ny - 1 : cy1;
        cz1 = cz1 >= cfg.nz ? cfg.nz - 1 : cz1;

        for (int cz = cz0; cz <= cz1; cz++)
            for (int cy = cy0; cy <= cy1; cy++)
                for (int cx = cx0; cx <= cx1; cx++)
                    cells[(cz * cfg.ny + cy) * cfg.nx + cx].push_back(i);
    }

    cellStart.resize(numCells + 1);
    cellTris.clear();
    for (int c = 0; c < numCells; c++) {
        cellStart[c] = (int)cellTris.size();
        for (int idx : cells[c])
            cellTris.push_back(idx);
    }
    cellStart[numCells] = (int)cellTris.size();
}
