#include "util/cpp/triangle_mesh.h"
#include "core/conversions.h"

#include <CGAL/AABB_face_graph_triangle_primitive.h>
#include <CGAL/AABB_traits.h>
#include <CGAL/AABB_tree.h>
#include <CGAL/AABB_triangle_primitive.h>
#include <CGAL/Polygon_mesh_processing/compute_normal.h>
#include <CGAL/subdivision_method_3.h>

#include <boost/optional/optional_io.hpp>
#include <cmath>
#include <iostream>
#include <sstream>

#include <omp.h>

namespace tangent_images {
namespace mesh {

const K::Vector_3 Pt2Vec(const K::Point_3 &pt) {
  return K::Vector_3(pt.x(), pt.y(), pt.z());
}

const K::Point_3 Vec2Pt(const K::Vector_3 &vec) {
  return K::Point_3(vec.x(), vec.y(), vec.z());
}

const size_t MaxOfThreeValues(const float s, const float t, const float u) {
  if (s > t && s > u) { return 0; }
  if (t > s && t > u) { return 1; }
  if (u > s && u > t) { return 2; }
  return 0;
}

TriangleMesh::TriangleMesh() {}

std::ostream &operator<<(std::ostream &out, const TriangleMesh &data) {
  out << data.ToString();
  return out;
}

TriangleMesh::TriangleMesh(torch::Tensor pts, torch::Tensor faces) {
  const size_t num_pts   = pts.size(0);
  const size_t num_faces = faces.size(0);
  this->_BuildMesh(pts.data<float>(), num_pts, faces.data<int64_t>(),
                   num_faces);
}

TriangleMesh::TriangleMesh(const std::vector<float> &pts,
                           const std::vector<int64_t> &faces) {
  const size_t num_pts   = pts.size() / 3;
  const size_t num_faces = faces.size() / 3;
  this->_BuildMesh(pts.data(), num_pts, faces.data(), num_faces);
}

void TriangleMesh::_BuildMesh(const float *pts, const size_t num_pts,
                              const int64_t *faces, const size_t num_faces) {
  // Store the vertex descriptors in order of emplacement so we can quickly
  // index when adding faces
  std::vector<VertexDescriptor> vert_desc;
  for (size_t i = 0; i < num_pts; i++) {
    // Create a vertex
    K::Point_3 pt(pts[3 * i], pts[3 * i + 1], pts[3 * i + 2]);

    // Add the vertex to the TriangleMesh
    vert_desc.push_back(this->_AddVertex(pt));
  }

  // Add the faces to TriangleMesh
  for (size_t i = 0; i < num_faces; i++) {
    // Parse the vertex connectivity data
    VertexDescriptor v0 = vert_desc[faces[3 * i]];
    VertexDescriptor v1 = vert_desc[faces[3 * i + 1]];
    VertexDescriptor v2 = vert_desc[faces[3 * i + 2]];
    this->_AddFace(v0, v1, v2);

    // TODO: Catch out-of-bounds vertices. Will segfault currently.
  }
}

const float TriangleMesh::PointNorm(const K::Point_3 &pt) {
  return sqrt(pt.x() * pt.x() + pt.y() * pt.y() + pt.z() * pt.z());
}
const float TriangleMesh::VectorNorm(const K::Vector_3 &vec) {
  return sqrt(vec.x() * vec.x() + vec.y() * vec.y() + vec.z() * vec.z());
}

const std::vector<float> TriangleMesh::ComputeBarycentricCoordinates(
    const K::Point_3 &pt, const K::Point_3 &A, const K::Point_3 &B,
    const K::Point_3 &C) {
  // Compute barycentric coordinates
  // s --> A
  // t --> B
  // u --> C
  // Compute the necessary vectors
  const auto BA = B - A;
  const auto CA = C - A;
  const auto W  = pt - A;

  // Area of the parallelogram
  const auto N = CGAL::cross_product(BA, CA);

  // Compute the ratios of subtriangles
  const float u = CGAL::cross_product(W, CA) * N / (N * N);
  const float t = CGAL::cross_product(BA, W) * N / (N * N);
  const float s = 1 - t - u;
  return {s, u, t};
}

void TriangleMesh::NormalizePoints() {
  // Get the iterator range over the vertices
  auto vertices_range = this->_mesh.vertices();
  for (auto it = vertices_range.begin(); it != vertices_range.end(); it++) {
    // Currently it's not possible to modify a Point_3, so just create a new
    // one with the normalized values
    auto &pt             = this->_mesh.point(*it);
    const float inv_norm = 1.0 / PointNorm(pt);
    pt = K::Point_3(pt.x() * inv_norm, pt.y() * inv_norm, pt.z() * inv_norm);
  }
}

const std::string TriangleMesh::ToString() const {
  std::stringstream ss;
  ss << "TriangleMesh ";
  ss << "(Vertices: " << this->NumVertices() << ", ";
  ss << "Faces: " << this->NumFaces() << ")";
  return ss.str();
}

void TriangleMesh::_Add(const float val) {
  // Get the iterator range over the vertices
  auto vertex_range = this->_mesh.vertices();

  // Iterate across all vertices
  for (auto it = vertex_range.begin(); it != vertex_range.end(); it++) {
    auto &pt = this->_mesh.point(*it);
    pt       = K::Point_3(pt.x() + val, pt.y() + val, pt.z() + val);
  }
}

void TriangleMesh::_Scale(const float val) {
  // Get the iterator range over the vertices
  auto vertex_range = this->_mesh.vertices();

  // Iterate across all vertices
  for (auto it = vertex_range.begin(); it != vertex_range.end(); it++) {
    auto &pt = this->_mesh.point(*it);
    pt       = K::Point_3(pt.x() * val, pt.y() * val, pt.z() * val);
  }
}

const K::Point_3 TriangleMesh::_GetFaceBarycenter(
    const FaceDescriptor &fd) const {
  // Get the representative half-edge of the face
  auto rep_he_idx = this->_mesh.halfedge(fd);

  // Go around each vertex and accumulate the coordinates
  float accum_x = 0.0;
  float accum_y = 0.0;
  float accum_z = 0.0;
  size_t j      = 0;
  for (auto he_idx : halfedges_around_face(rep_he_idx, this->_mesh)) {
    auto vert_idx = target(he_idx, this->_mesh);
    auto pt       = this->_mesh.point(vert_idx);
    accum_x += pt.x();
    accum_y += pt.y();
    accum_z += pt.z();
    j++;
  }
  return K::Point_3(accum_x / j, accum_y / j, accum_z / j);
}

const std::vector<int64_t> TriangleMesh::_GetVerticesAdjacentToFace(
    const FaceDescriptor &fd) const {
  // Initialize vector to hold descriptors
  std::vector<int64_t> adj_vertices;

  // Get the representative half-edge of the face
  auto rep_he_idx = this->_mesh.halfedge(fd);

  // Iterate over the half-edges of this face
  for (auto he_idx : halfedges_around_face(rep_he_idx, this->_mesh)) {
    const auto vert_idx = target(he_idx, this->_mesh);
    adj_vertices.push_back(static_cast<int64_t>(vert_idx));
  }

  return adj_vertices;
}

const std::vector<int64_t> TriangleMesh::_GetFacesAdjacentToFace(
    const FaceDescriptor &fd) const {
  // Get the representative half-edge of the face
  auto rep_he_idx = this->_mesh.halfedge(fd);

  // Initialize vector to hold descriptors
  std::vector<int64_t> adj_face_desc;

  // Iterate over the half-edges of this face. The adjacent faces can be
  // found as the face indices associates with the opposite half-edges
  for (auto he_idx : halfedges_around_face(rep_he_idx, this->_mesh)) {
    auto opp_he_idx  = this->_mesh.opposite(he_idx);  // Opposite half-edge
    int64_t adj_fidx = -1;  // Default for border cases

    // Only get the adjacent face ID if not on the border
    if (!(this->_mesh.is_border(opp_he_idx))) {
      adj_fidx = static_cast<int64_t>(this->_mesh.face(opp_he_idx));
    }
    adj_face_desc.push_back(adj_fidx);
  }
  return adj_face_desc;
}

const std::vector<int64_t> TriangleMesh::_GetVerticesAdjacentToVertex(
    const VertexDescriptor &vd) const {
  std::vector<int64_t> indices;

  // An outgoing half-edge to vertex vd
  auto he    = this->_mesh.opposite(this->_mesh.halfedge(vd));
  auto start = he;
  do {
    // Add the vertex being pointed to to the vector
    indices.push_back(static_cast<int64_t>(this->_mesh.target(he)));

    // Follow the opposite half-edge back to the source vertex and then go to
    // the next outgoing half edge
    he = this->_mesh.next(this->_mesh.opposite(he));
  } while (he != start);

  return indices;
}

const std::vector<int64_t> TriangleMesh::_GetFacesAdjacentToVertex(
    const VertexDescriptor &vd) const {
  std::vector<int64_t> indices;

  // An outgoing half-edge to vertex vd
  auto he    = this->_mesh.opposite(this->_mesh.halfedge(vd));
  auto start = he;
  do {
    // Add the face associated with this half-edge to the vector
    indices.push_back(static_cast<int64_t>(this->_mesh.face(he)));

    // Follow the opposite half-edge back to the source vertex and then go to
    // the next outgoing half edge
    he = this->_mesh.next(this->_mesh.opposite(he));
  } while (he != start);

  return indices;
}

const VertexDescriptor TriangleMesh::_AddVertex(const K::Point_3 &pt) {
  return this->_mesh.add_vertex(pt);
}

const FaceDescriptor TriangleMesh::_AddFace(const VertexDescriptor &v0,
                                            const VertexDescriptor &v1,
                                            const VertexDescriptor &v2) {
  return this->_mesh.add_face(v0, v1, v2);
}

// Returns V x 3
const torch::Tensor TriangleMesh::GetVertices() const {
  torch::Tensor tensor_vertices = torch::zeros(
      {static_cast<int64_t>(this->NumVertices()), 3}, torch::kFloat);

  // The pointer representation is for OpenMP compatibility (TODO)
  float *tensor_vertices_ptr = tensor_vertices.data<float>();

  // Get the iterator range over the vertices
  auto vertices_range = this->_mesh.vertices();
  for (auto it = vertices_range.begin(); it != vertices_range.end(); it++) {
    size_t vidx                       = static_cast<size_t>(*it);
    auto pt                           = this->_mesh.point(*it);
    tensor_vertices_ptr[3 * vidx]     = pt.x();
    tensor_vertices_ptr[3 * vidx + 1] = pt.y();
    tensor_vertices_ptr[3 * vidx + 2] = pt.z();
  }
  return tensor_vertices;
}

// Returns F x 3 x 3, F sets of 3 rows of points
const torch::Tensor TriangleMesh::GetAllFaceVertexCoordinates() const {
  const size_t num_faces = this->NumFaces();
  torch::Tensor tensor_coords =
      torch::zeros({static_cast<int64_t>(num_faces), 3, 3}, torch::kFloat);

  // Get the iterator range over the vertices
  auto faces_range = this->_mesh.faces();

  float *tensor_coords_ptr = tensor_coords.data<float>();
  for (auto it = faces_range.begin(); it != faces_range.end(); it++) {
    // Face index
    size_t fidx = static_cast<size_t>(*it);

    // Get the representative half-edge of the face
    auto rep_he_idx = this->_mesh.halfedge(*it);

    int j = 0;  // To count which vertex we're on
    for (auto he_idx : halfedges_around_face(rep_he_idx, this->_mesh)) {
      auto vert_idx                           = target(he_idx, this->_mesh);
      auto pt                                 = this->_mesh.point(vert_idx);
      tensor_coords_ptr[9 * fidx + 3 * j]     = pt.x();
      tensor_coords_ptr[9 * fidx + 3 * j + 1] = pt.y();
      tensor_coords_ptr[9 * fidx + 3 * j + 2] = pt.z();
      j++;
    }
  }

  return tensor_coords;
}

// // Returns F x 3, F rows of 3 indices
const torch::Tensor TriangleMesh::GetAllFaceVertexIndices() const {
  const size_t num_faces = this->NumFaces();
  torch::Tensor tensor_indices =
      torch::zeros({static_cast<int64_t>(num_faces), 3}, torch::kLong);

  // Get the iterator range over the vertices
  auto faces_range = this->_mesh.faces();

  int64_t *tensor_indices_ptr = tensor_indices.data<int64_t>();
  for (auto it = faces_range.begin(); it != faces_range.end(); it++) {
    // Face index
    size_t fidx = static_cast<size_t>(*it);

    // Get the representative half-edge of the face
    auto rep_he_idx = this->_mesh.halfedge(*it);

    int j = 0;  // To count which vertex we're on
    for (auto he_idx : halfedges_around_face(rep_he_idx, this->_mesh)) {
      auto vert_idx                    = target(he_idx, this->_mesh);
      tensor_indices_ptr[3 * fidx + j] = static_cast<int64_t>(vert_idx);
      j++;
    }
  }
  return tensor_indices;
}

// Returns F x 3 x 3, F sets of 3 rows of points
const torch::Tensor TriangleMesh::GetFaceBarycenters() const {
  const size_t num_faces = this->NumFaces();
  torch::Tensor tensor_barycenters =
      torch::zeros({static_cast<int64_t>(num_faces), 3}, torch::kFloat);

  // Get the iterator range over the vertices
  auto faces_range = this->_mesh.faces();

  // Iterate across all faces
  float *tensor_barycenters_ptr = tensor_barycenters.data<float>();
  for (auto it = faces_range.begin(); it != faces_range.end(); it++) {
    // Compute the barycenter for the face
    K::Point_3 barycenter = this->_GetFaceBarycenter(*it);

    // Face index
    size_t fidx = static_cast<size_t>(*it);

    // Add the point to the output tensor
    tensor_barycenters_ptr[3 * fidx]     = barycenter.x();
    tensor_barycenters_ptr[3 * fidx + 1] = barycenter.y();
    tensor_barycenters_ptr[3 * fidx + 2] = barycenter.z();
  }

  return tensor_barycenters;
}

const torch::Tensor TriangleMesh::GetAdjacentFaceIndicesToFaces() const {
  torch::Tensor tensor_indices =
      torch::zeros({static_cast<int64_t>(this->NumFaces()), 3}, torch::kLong);

  int64_t *tensor_indices_ptr = tensor_indices.data<int64_t>();

  // Get the iterator range over the vertices
  auto faces_range = this->_mesh.faces();

  // Iterate across all faces
  size_t i = 0;
  for (auto it = faces_range.begin(); it != faces_range.end(); it++) {
    auto face_indices = this->_GetFacesAdjacentToFace(*it);
    std::copy(face_indices.data(), face_indices.data() + 3,
              tensor_indices_ptr + 3 * i);
    i++;
  }

  return tensor_indices;
}

const std::map<int64_t, torch::Tensor>
TriangleMesh::GetAdjacentVertexIndicesToVertices() const {
  // Map to hold output dictionary
  std::map<int64_t, torch::Tensor> dict;

  // Get the iterator range over the vertices
  auto vertex_range = this->_mesh.vertices();

  // Iterate across all vertices
  for (auto it = vertex_range.begin(); it != vertex_range.end(); it++) {
    // Get adjacent vertices
    auto vertex_indices = this->_GetVerticesAdjacentToVertex(*it);

    // Copy the data to a torch tensor
    torch::Tensor tensor_indices = torch::zeros(
        {static_cast<int64_t>(vertex_indices.size())}, torch::kLong);
    std::copy(vertex_indices.data(),
              vertex_indices.data() + vertex_indices.size(),
              tensor_indices.data<int64_t>());

    dict.emplace(static_cast<int64_t>(*it), tensor_indices);
  }

  return dict;
}

const std::map<int64_t, torch::Tensor>
TriangleMesh::GetAdjacentFaceIndicesToVertices() const {
  // Map to hold output dictionary
  std::map<int64_t, torch::Tensor> dict;

  // Get the iterator range over the vertices
  auto vertex_range = this->_mesh.vertices();

  // Iterate across all vertices
  for (auto it = vertex_range.begin(); it != vertex_range.end(); it++) {
    // Get adjacent vertices
    auto face_indices = this->_GetFacesAdjacentToVertex(*it);

    // Copy th data to a torch tensor
    torch::Tensor tensor_indices = torch::zeros(
        {static_cast<int64_t>(face_indices.size())}, torch::kLong);
    std::copy(face_indices.data(), face_indices.data() + face_indices.size(),
              tensor_indices.data<int64_t>());

    dict.emplace(static_cast<int64_t>(*it), tensor_indices);
  }

  return dict;
}

const torch::Tensor TriangleMesh::GetFacesAdjacentToFace(
    const size_t face_idx) const {
  const auto adj_faces =
      this->_GetFacesAdjacentToFace(FaceDescriptor(face_idx));
  torch::Tensor tensor_indices =
      torch::zeros({static_cast<int64_t>(adj_faces.size())}, torch::kLong);
  std::copy(adj_faces.data(), adj_faces.data() + adj_faces.size(),
            tensor_indices.data<int64_t>());
  return tensor_indices;
}

const torch::Tensor TriangleMesh::GetVerticesAdjacentToFace(
    const size_t face_idx) const {
  const auto adj_vertices =
      this->_GetVerticesAdjacentToFace(FaceDescriptor(face_idx));
  torch::Tensor tensor_indices =
      torch::zeros({static_cast<int64_t>(adj_vertices.size())}, torch::kLong);
  std::copy(adj_vertices.data(), adj_vertices.data() + adj_vertices.size(),
            tensor_indices.data<int64_t>());
  return tensor_indices;
}

const torch::Tensor TriangleMesh::GetFacesAdjacentToVertex(
    const size_t vertex_idx) const {
  const auto adj_faces =
      this->_GetFacesAdjacentToVertex(VertexDescriptor(vertex_idx));
  torch::Tensor tensor_indices =
      torch::zeros({static_cast<int64_t>(adj_faces.size())}, torch::kLong);
  std::copy(adj_faces.data(), adj_faces.data() + adj_faces.size(),
            tensor_indices.data<int64_t>());
  return tensor_indices;
}

const torch::Tensor TriangleMesh::GetVerticesAdjacentToVertex(
    const size_t vertex_idx) const {
  const auto adj_vertices =
      this->_GetVerticesAdjacentToVertex(VertexDescriptor(vertex_idx));
  torch::Tensor tensor_indices =
      torch::zeros({static_cast<int64_t>(adj_vertices.size())}, torch::kLong);
  std::copy(adj_vertices.data(), adj_vertices.data() + adj_vertices.size(),
            tensor_indices.data<int64_t>());
  return tensor_indices;
}

void TriangleMesh::LoopSubdivide(const size_t order) {
  CGAL::Subdivision_method_3::Loop_subdivision(
      this->_mesh,
      CGAL::Subdivision_method_3::parameters::number_of_iterations(order));
}

void TriangleMesh::CatmullClarkSubdivide(const size_t order) {
  CGAL::Subdivision_method_3::CatmullClark_subdivision(
      this->_mesh,
      CGAL::Subdivision_method_3::parameters::number_of_iterations(order));
}

void TriangleMesh::MidpointSubdivide(const size_t order) {
  CGAL::Subdivision_method_3::PTQ(
      this->_mesh, MidpointSubdivisionMask(this->_mesh),
      CGAL::Subdivision_method_3::parameters::number_of_iterations(order));
}

const std::vector<torch::Tensor> TriangleMesh::ComputeNormals() {
  // From the example given here:
  // https://doc.cgal.org/latest/Polygon_mesh_processing/Polygon_mesh_processing_2compute_normals_example_8cpp-example.html

  // Add the propery maps to the mesh for the normal computation
  auto fnormals =
      this->_mesh
          .add_property_map<boost::graph_traits<SurfaceMesh>::face_descriptor,
                            K::Vector_3>("f:normals", CGAL::NULL_VECTOR)
          .first;
  auto vnormals = this->_mesh
                      .add_property_map<
                          boost::graph_traits<SurfaceMesh>::vertex_descriptor,
                          K::Vector_3>("v:normals", CGAL::NULL_VECTOR)
                      .first;

  // Computes both the vertex and face normals
  CGAL::Polygon_mesh_processing::compute_normals(
      this->_mesh, vnormals, fnormals,
      CGAL::Polygon_mesh_processing::parameters::vertex_point_map(
          this->_mesh.points()));

  // Create output torch tensors
  torch::Tensor tensor_vert_normals = torch::zeros(
      {static_cast<int64_t>(this->NumVertices()), 3}, torch::kFloat);
  torch::Tensor tensor_face_normals =
      torch::zeros({static_cast<int64_t>(this->NumFaces()), 3}, torch::kFloat);
  float *tensor_vert_normals_ptr = tensor_vert_normals.data<float>();
  float *tensor_face_normals_ptr = tensor_face_normals.data<float>();

  // Copy the normals to torch tensor form
  size_t i = 0;
  for (const auto vd : this->_mesh.vertices()) {
    auto vnorm                         = vnormals[vd];
    tensor_vert_normals_ptr[3 * i]     = vnorm.x();
    tensor_vert_normals_ptr[3 * i + 1] = vnorm.y();
    tensor_vert_normals_ptr[3 * i + 2] = vnorm.z();
    i++;
  }
  i = 0;
  for (const auto fd : this->_mesh.faces()) {
    auto fnorm                         = fnormals[fd];
    tensor_face_normals_ptr[3 * i]     = fnorm.x();
    tensor_face_normals_ptr[3 * i + 1] = fnorm.y();
    tensor_face_normals_ptr[3 * i + 2] = fnorm.z();
    i++;
  }

  return {tensor_vert_normals, tensor_face_normals};
}

const float TriangleMesh::SpheroidRadius() const {
  // Compute the radius as the max norm of the points on the icosphere
  float radius = 0.0;

  // Get the iterator range over the vertices
  const auto vertices_range = this->_mesh.vertices();

  // Iterate over all vertices in the mesh sotring the max norms
  for (auto it = vertices_range.begin(); it != vertices_range.end(); it++) {
    const float cur_radius = this->PointNorm(this->_mesh.point(*it));
    if (cur_radius > radius) { radius = cur_radius; }
  }
  // Return the max radius
  return radius;
}

const float TriangleMesh::GetVertexResolution() const {
  float res = 0.;

  // Get the iterator range over the vertices
  auto vertices_range = this->_mesh.vertices();

  // Iterate over all vertices in the mesh
  for (auto it = vertices_range.begin(); it != vertices_range.end(); it++) {
    // Get the list of vertices that are adjacent to this one
    const auto adj_vert = this->_GetVerticesAdjacentToVertex(*it);

    // Compute the mean distance between each adjacent vertex and this one
    float acc_norm     = 0.;
    int num_adj        = 0;
    const auto &cur_pt = this->_mesh.point(*it);
    for (const auto adj_idx : adj_vert) {
      auto &adj_pt = this->_mesh.point(VertexDescriptor(adj_idx));

      // Distance between current point and adjacent point
      acc_norm += PointNorm(Vec2Pt(cur_pt - adj_pt));
      num_adj++;
    }
    res += acc_norm / float(num_adj);
  }
  return res / float(this->NumVertices());
}

const float TriangleMesh::GetAngularResolution() const {
  float res = 0.;

  // Get the iterator range over the vertices
  auto vertices_range = this->_mesh.vertices();

  // Iterate over all vertices in the mesh
  for (auto it = vertices_range.begin(); it != vertices_range.end(); it++) {
    // Get the list of vertices that are adjacent to this one
    const auto adj_vert = this->_GetVerticesAdjacentToVertex(*it);

    // Compute the mean angle between each adjacent vertex and this one
    float acc_angle    = 0.;
    int num_adj        = 0;
    const auto &cur_pt = this->_mesh.point(*it);
    for (const auto adj_idx : adj_vert) {
      auto &adj_pt = this->_mesh.point(VertexDescriptor(adj_idx));

      // Angle between current point and adjacent point (radians)
      acc_angle += std::acos((Pt2Vec(cur_pt) * Pt2Vec(adj_pt)) /
                             (PointNorm(cur_pt) * PointNorm(adj_pt)));
      num_adj++;
    }
    res += acc_angle / float(num_adj);
  }
  return res / float(this->NumVertices());
}

const std::vector<torch::Tensor> TriangleMesh::GetIcosphereConvolutionOperator(
    torch::Tensor samples, const TriangleMesh &icosphere, const bool keepdim,
    const bool nearest) {
  // Parse the dimensions
  const size_t samples_height = samples.size(0);
  const size_t samples_width  = samples.size(1);

  // (Sample maps with only 3 dimensions are used for resampling and thus
  // have a "kernel size" of 1)
  const size_t kernel_size = (samples.dim() == 3) ? 1 : samples.size(2);

  // Create the output tensors
  torch::Tensor tensor_face_indices;
  torch::Tensor tensor_vertex_indices;
  torch::Tensor tensor_weights;

  // Shape of output tensors are based on some arguments
  if (kernel_size > 1 || (kernel_size == 1 && keepdim)) {
    tensor_face_indices = torch::zeros({static_cast<int64_t>(samples_height),
                                        static_cast<int64_t>(samples_width),
                                        static_cast<int64_t>(kernel_size)},
                                       torch::kLong);
    tensor_vertex_indices =
        torch::zeros({static_cast<int64_t>(samples_height),
                      static_cast<int64_t>(samples_width),
                      static_cast<int64_t>(kernel_size), 3},
                     torch::kLong);
    tensor_weights = torch::zeros({static_cast<int64_t>(samples_height),
                                   static_cast<int64_t>(samples_width),
                                   static_cast<int64_t>(kernel_size), 3},
                                  torch::kFloat);
  } else {
    tensor_face_indices = torch::zeros({static_cast<int64_t>(samples_height),
                                        static_cast<int64_t>(samples_width)},
                                       torch::kLong);
    tensor_vertex_indices =
        torch::zeros({static_cast<int64_t>(samples_height),
                      static_cast<int64_t>(samples_width), 3},
                     torch::kLong);
    tensor_weights = torch::zeros({static_cast<int64_t>(samples_height),
                                   static_cast<int64_t>(samples_width), 3},
                                  torch::kFloat);
  }

  // Get the pointers to the tensors
  const auto samples_ptr         = samples.data<float>();
  auto tensor_face_indices_ptr   = tensor_face_indices.data<int64_t>();
  auto tensor_vertex_indices_ptr = tensor_vertex_indices.data<int64_t>();
  auto tensor_weights_ptr        = tensor_weights.data<float>();

  // Construct AABB tree and computes internal KD-tree data structure to
  // accelerate distance queries. Crazy nested templates to avoid excessive
  // CGAL typedefs. Examples with typedefs are found at
  // https://doc.cgal.org/latest/AABB_tree/index.html
  CGAL::AABB_tree<CGAL::AABB_traits<
      K, CGAL::AABB_face_graph_triangle_primitive<SurfaceMesh>>>
      tree(faces(icosphere._mesh).first, faces(icosphere._mesh).second,
           icosphere._mesh);
  tree.accelerate_distance_queries();

  // Go through every sample in the <samples> tensor
  // These are independent calculations, so we can parallelize it with OpenMP
  size_t i;
#pragma omp parallel for shared(                        \
    tensor_face_indices_ptr, tensor_vertex_indices_ptr, \
    tensor_weights_ptr) private(i) schedule(static)
  for (i = 0; i < samples_height * samples_width; i++) {
    // Current kernel sample set
    size_t cur_kernel = i * kernel_size * 2;

    // Go through each element in the kernel and find its intersection on the
    // mesh
    for (size_t j = 0; j < kernel_size; j++) {
      // Convert the spherical coordinate to XYZ coordinates
      float x, y, z;
      core::SphericalToXYZ(samples_ptr[cur_kernel + j * 2],
                           samples_ptr[cur_kernel + j * 2 + 1], x, y, z);

      // Create a ray to query
      auto pt = K::Point_3(x, y, z);
      K::Ray_3 ray_query(CGAL::ORIGIN, pt);

      // Compute the ray-mesh intersection
      // Note that this function returns a boost::optional type which is why
      // there's a trailing `get()`
      auto intersection = tree.first_intersection(ray_query).get();

      // Retrieve the intersection point and the face ID of the intersection
      // Within the boost::optional return type is a boost::variant, hence
      // why we must call another `get<type>()` to get the value. CGAL is
      // great, but super convoluted...
      const K::Point_3 intersection_pt =
          boost::get<K::Point_3>(intersection.first);
      const FaceDescriptor fd =
          boost::get<FaceDescriptor>(intersection.second);

      // Copy the found face index to the output tensor
      tensor_face_indices_ptr[i * kernel_size + j] = static_cast<int64_t>(fd);

      // Copy the vertex indices to the output tensor
      auto vertices = icosphere._GetVerticesAdjacentToFace(fd);
      tensor_vertex_indices_ptr[i * kernel_size * 3 + j * 3]     = vertices[0];
      tensor_vertex_indices_ptr[i * kernel_size * 3 + j * 3 + 1] = vertices[1];
      tensor_vertex_indices_ptr[i * kernel_size * 3 + j * 3 + 2] = vertices[2];

      // Compute the barycentric weights on the triangle
      auto bary_coords = TriangleMesh::ComputeBarycentricCoordinates(
          intersection_pt,
          icosphere._mesh.point(VertexDescriptor(vertices[0])),
          icosphere._mesh.point(VertexDescriptor(vertices[1])),
          icosphere._mesh.point(VertexDescriptor(vertices[2])));

      // Copy the barycentric weights to the output tensor
      if (nearest) {
        const size_t maxVal =
            MaxOfThreeValues(bary_coords[0], bary_coords[1], bary_coords[2]);
        tensor_weights_ptr[i * kernel_size * 3 + j * 3 + 0] =
            maxVal == 0 ? 1 : 0;
        tensor_weights_ptr[i * kernel_size * 3 + j * 3 + 1] =
            maxVal == 1 ? 1 : 0;
        tensor_weights_ptr[i * kernel_size * 3 + j * 3 + 2] =
            maxVal == 2 ? 1 : 0;
      } else {
        std::copy(bary_coords.data(), bary_coords.data() + 3,
                  tensor_weights_ptr + i * kernel_size * 3 + j * 3);
      }
    }
  }
  return {tensor_face_indices, tensor_vertex_indices, tensor_weights};
}

const float ComputeDistortionMagnitude(const float x, const float y,
                                       const std::vector<float> &radial,
                                       const std::vector<float> &tangential) {
  const float k1     = radial[0];
  const float k2     = radial[1];
  const float k3     = radial[2];
  const float t1     = tangential[0];
  const float t2     = tangential[1];
  const float r2     = x * x + y * y;
  const float r4     = r2 * r2;
  const float r6     = r4 * r2;
  const float k_diff = (k1 * r2 + k2 * r4 + k3 * r6);
  const float t_x    = t2 * (r2 + 2 * x * x) + 2 * t1 * x * y;
  const float t_y    = t1 * (r2 + 2 * y * y) + 2 * t2 * x * y;
  const float x_off  = x * k_diff + t_x;
  const float y_off  = y * k_diff + t_y;
  return std::sqrt(x_off * x_off + y_off * y_off);
}

const torch::Tensor TriangleMesh::DistortImageGrid(
    TriangleMesh &image_grid, const std::vector<float> &radial,
    const std::vector<float> tangential) {
  torch::Tensor new_vertices = torch::zeros(
      {static_cast<int64_t>(image_grid.NumVertices()), 3}, torch::kFloat);

  // Get the iterator range over the vertices
  auto vertices_range = image_grid._mesh.vertices();
  for (auto it = vertices_range.begin(); it != vertices_range.end(); it++) {
    // Currently it's not possible to modify a Point_3, so just create a
    // new one with the normalized values
    auto &pt = image_grid._mesh.point(*it);
    const float dist_mag =
        ComputeDistortionMagnitude(pt.x(), pt.y(), radial, tangential);
    pt = K::Point_3(pt.x(), pt.y(), dist_mag);
  }
  return new_vertices;
}

const torch::Tensor TriangleMesh::GetFaceTuples(const size_t order) {
  // Create icosphere
  TriangleMesh icosphere = GenerateIcosphere(order);

  // Output tensor
  torch::Tensor tuples = torch::zeros(
      {static_cast<int64_t>(icosphere.NumFaces() / 2), 2}, torch::kLong);
  auto tuples_ptr = tuples.data<int64_t>();

  // Index counter
  size_t tuple_count = 0;

  // Go through each halfedge emanating from the 0-th vertex as each one of
  // these defines the starting point of a "strip" on the net of the
  // icosahedron
  for (auto he :
       CGAL::halfedges_around_source(VertexDescriptor(0), icosphere._mesh)) {
    // For each subdivided strip within the main strip
    for (size_t i = 0; i < std::pow(2, order); i++) {
      // Reference the current half edge moving down the strip of the
      // icosahedron
      auto he_strip = he;

      // Go through each tuple in the strip
      for (size_t k = 0; k < std::pow(2, order + 1); k++) {
        // Grab the face tuple
        const auto f0 = icosphere._mesh.face(he_strip);
        const auto f1 = icosphere._mesh.face(
            icosphere._mesh.opposite(icosphere._mesh.next(he_strip)));

        // Add the tuple to the tensor
        tuples_ptr[2 * tuple_count]     = static_cast<int64_t>(f0);
        tuples_ptr[2 * tuple_count + 1] = static_cast<int64_t>(f1);

        // Increment tuple counter
        tuple_count++;

        // Advance the halfedge down the strip
        he_strip =
            icosphere._mesh.next(icosphere._mesh.opposite(icosphere._mesh.next(
                icosphere._mesh.opposite(icosphere._mesh.next(he_strip)))));
      }
      // Move the halfedge to the next strip
      he = icosphere._mesh.opposite(icosphere._mesh.next(icosphere._mesh.next(
          icosphere._mesh.opposite(icosphere._mesh.next(he)))));
    }
  }

  return tuples;
}

void IntersectionToUV(const K::Point_3 &intersection, const K::Point_3 &point0,
                      const K::Point_3 &point1, const K::Point_3 &point2,
                      float &tex_u, float &tex_v) {
  // U-coordinate triangle conversion
  const auto vec    = intersection - point0;  // Vector to intersection
  const auto v_axis = point2 - point0;        // Vector down the +V axis
  const auto u_axis = point1 - point0;        // Opposite height vector
  tex_u =
      (vec * u_axis) / u_axis.squared_length();  // Normalized proj. on +U axis
  tex_v =
      (vec * v_axis) / v_axis.squared_length();  // Normalized proj. on +V axis
}

// The points are stored in a flat array of doubles
// The triangles are stored in a flat array of indices
// referring to an array of coordinates: three consecutive
// coordinates represent a point, and three consecutive
// indices represent a triangle.
typedef size_t *Point_index_iterator;
// Let us now define the iterator on triangles that the tree needs:
class Triangle_iterator
    : public boost::iterator_adaptor<
          Triangle_iterator,               // Derived
          Point_index_iterator,            // Base
          boost::use_default,              // Value
          boost::forward_traversal_tag> {  // CategoryOrTraversal

 public:
  Triangle_iterator() : Triangle_iterator::iterator_adaptor_() {}
  explicit Triangle_iterator(Point_index_iterator p)
      : Triangle_iterator::iterator_adaptor_(p) {}

 private:
  friend class boost::iterator_core_access;
  void increment() { this->base_reference() += 3; }
};
// The following primitive provides the conversion facilities between
// my own triangle and point types and the CGAL ones
struct My_triangle_primitive {
 public:
  typedef Triangle_iterator Id;
  // the CGAL types returned
  typedef K::Point_3 Point;
  typedef K::Triangle_3 Datum;
  // a static pointer to the vector containing the points
  // is needed to build the triangles on the fly:
  static const float *point_container;

 private:
  Id m_it;  // this is what the AABB tree stores internally
 public:
  My_triangle_primitive() {}  // default constructor needed
  // the following constructor is the one that receives the iterators from the
  // iterator range given as input to the AABB_tree
  My_triangle_primitive(Triangle_iterator a) : m_it(a) {}
  Id id() const { return m_it; }
  // on the fly conversion from the internal data to the CGAL types
  Datum datum() const {
    Point_index_iterator p_it = m_it.base();
    Point p(*(point_container + 3 * (*p_it)),
            *(point_container + 3 * (*p_it) + 1),
            *(point_container + 3 * (*p_it) + 2));
    ++p_it;
    Point q(*(point_container + 3 * (*p_it)),
            *(point_container + 3 * (*p_it) + 1),
            *(point_container + 3 * (*p_it) + 2));
    ++p_it;
    Point r(*(point_container + 3 * (*p_it)),
            *(point_container + 3 * (*p_it) + 1),
            *(point_container + 3 * (*p_it) + 2));
    return Datum(p, q, r);  // assembles triangle from three points
  }
  // one point which must be on the primitive
  Point reference_point() const {
    return Point(*(point_container + 3 * (*m_it)),
                 *(point_container + 3 * (*m_it) + 1),
                 *(point_container + 3 * (*m_it) + 2));
  }
};
const float *My_triangle_primitive::point_container = 0;

std::vector<torch::Tensor> FindTangentPlaneIntersections(
    torch::Tensor plane_corners, torch::Tensor rays) {
  typedef CGAL::AABB_traits<K, My_triangle_primitive> My_AABB_traits;
  typedef CGAL::AABB_tree<My_AABB_traits> Tree;

  // Sizes
  const size_t num_quads = plane_corners.size(0);
  const size_t num_rays  = rays.size(0);

  // Input tensor pointers
  const auto *plane_corners_ptr = plane_corners.data<float>();
  const auto *rays_ptr          = rays.data<float>();

  // Output tensors
  torch::Tensor quad_indices =
      torch::zeros({static_cast<int64_t>(num_rays)}, torch::kLong);
  torch::Tensor uv_coords =
      torch::zeros({static_cast<int64_t>(num_rays), 2}, torch::kFloat);
  auto quad_indices_ptr = quad_indices.data<int64_t>();
  auto uv_coords_ptr    = uv_coords.data<float>();

  // Create all tangent planes as two triangles. We expect plane_corners to
  // be an N x 4 x 3 tensor with corners as in the image below, where both
  // triangles' normals are coming out of the screen.
  //
  //  a     b
  //  * ---- *
  //  | \ O |    ^
  //  |  \  |    | Up-oriented direction
  //  | E \ |    |
  //  * ---- *
  //  c     d
  //
  // Each triangle tuple is adjacent in the vector so that odd/even
  // triangles are tracked accordingly and we can get the face index back.
  My_triangle_primitive::point_container = plane_corners_ptr;
  size_t triangles[6 * num_quads];
  for (size_t i = 0; i < num_quads; i++) {
    // Create the face corners
    triangles[6 * i]     = 4 * i;
    triangles[6 * i + 1] = 4 * i + 2;
    triangles[6 * i + 2] = 4 * i + 3;
    triangles[6 * i + 3] = 4 * i + 3;
    triangles[6 * i + 4] = 4 * i + 1;
    triangles[6 * i + 5] = 4 * i;
  }

  // Build the AABB tree
  const Tree tree(Triangle_iterator(triangles),
                  Triangle_iterator(triangles + 6 * num_quads));
  tree.accelerate_distance_queries();

  // For all rays, find the intersection with all triangles. Expects rays
  // to be defined in spherical coordinates in an N x 2 tensor
  // For some reason an OMP par-for loop segfaults occasionally when performing
  // a ray query on the tree...
  size_t i;
#pragma omp parallel for shared(quad_indices_ptr, uv_coords_ptr) private(i) \
    schedule(static)
  for (i = 0; i < num_rays; i++) {
    // Convert the spherical coordinate to XYZ coordinates
    float x, y, z;
    core::SphericalToXYZ(rays_ptr[2 * i], rays_ptr[2 * i + 1], x, y, z);

    // Create a ray to query
    const auto pt = K::Point_3(x, y, z);
    const K::Ray_3 ray_query(CGAL::ORIGIN, pt);

    // Compute the ray-face intersection
    // Note that this function returns a boost::optional type which is why
    // there's a trailing `get()`
    const auto intersection_query = tree.first_intersection(ray_query);
    const auto intersection       = intersection_query.get();

    // Retrieve the intersection point and the face ID of the intersection
    // Within the boost::optional return type is a boost::variant, hence
    // why we must call another `get<type>()` to get the value. CGAL is
    // great, but super convoluted...
    // The first part of the pair is the actual point of intersection
    const K::Point_3 intersection_pt =
        boost::get<K::Point_3>(intersection.first);
    // The second part of the pair is the reference point idx of the
    // intersected face. <face_pt_idx> / 4 gives the "quad" index because the
    // first face of each quad is referenced by its point ID and each quad has
    // 4 points
    const size_t face_pt_idx = boost::get<size_t>(*intersection.second);
    const size_t quad_idx    = face_pt_idx / 4;

    // Get point indices (ordered in rows, so sequential)
    const size_t point_idx0 = triangles[6 * quad_idx];
    const size_t point_idx1 = point_idx0 + 1;
    const size_t point_idx2 = point_idx0 + 2;

    const K::Point_3 point0 =
        K::Point_3(plane_corners_ptr[point_idx0 * 3],
                   plane_corners_ptr[point_idx0 * 3 + 1],
                   plane_corners_ptr[point_idx0 * 3 + 2]);

    const K::Point_3 point1 =
        K::Point_3(plane_corners_ptr[point_idx1 * 3],
                   plane_corners_ptr[point_idx1 * 3 + 1],
                   plane_corners_ptr[point_idx1 * 3 + 2]);

    const K::Point_3 point2 =
        K::Point_3(plane_corners_ptr[point_idx2 * 3],
                   plane_corners_ptr[point_idx2 * 3 + 1],
                   plane_corners_ptr[point_idx2 * 3 + 2]);

    // Convert to UV texture coordinates on a single plane
    float u, v;
    IntersectionToUV(intersection_pt, point0, point1, point2, u, v);

    // Return plane index(e.g.quad face index aka tangent plane index in
    // tensor) and UV coords within the plane
    quad_indices_ptr[i]      = quad_idx;
    uv_coords_ptr[2 * i]     = u;
    uv_coords_ptr[2 * i + 1] = v;
  }

  return {quad_indices, uv_coords};
}

// kp_3d: M x 3
// kp_quad: M
// kp_desc: M x 128
// kp_scale: M
// kp_orient: M
// plane_corners: N x 4 x 3
// returns: {M x 3 valid 3D keypoints, M x 128 valid descriptors, M scales, M
// orientations}
std::vector<torch::Tensor> FindVisibleKeypoints(torch::Tensor kp_3d,
                                                torch::Tensor kp_quad,
                                                torch::Tensor kp_desc,
                                                torch::Tensor kp_scale,
                                                torch::Tensor kp_orient,
                                                torch::Tensor plane_corners) {
  typedef CGAL::AABB_traits<K, My_triangle_primitive> My_AABB_traits;
  typedef CGAL::AABB_tree<My_AABB_traits> Tree;

  // -------------------------------------------
  // First create the tangent plane polygons
  // -------------------------------------------

  // Sizes
  const size_t num_quads = plane_corners.size(0);
  const size_t num_kp    = kp_3d.size(0);

  // Input tensor pointers
  const auto *plane_corners_ptr = plane_corners.data<float>();
  const auto *kp_3d_ptr         = kp_3d.data<float>();
  const auto *kp_quad_ptr       = kp_quad.data<int64_t>();
  const auto *kp_desc_ptr       = kp_desc.data<float>();
  const auto *kp_scale_ptr      = kp_scale.data<float>();
  const auto *kp_orient_ptr     = kp_orient.data<float>();

  // Create all tangent planes as two triangles. We expect plane_corners to
  // be an N x 4 x 3 tensor with corners as in the image below, where both
  // triangles' normals are coming out of the screen.
  //
  //  a     b
  //  * ---- *
  //  | \ O |    ^
  //  |  \  |    | Up-oriented direction
  //  | E \ |    |
  //  * ---- *
  //  c     d
  //
  // Each triangle tuple is adjacent in the vector so that odd/even
  // triangles are tracked accordingly and we can get the face index back.
  My_triangle_primitive::point_container = plane_corners_ptr;
  size_t triangles[6 * num_quads];
  for (size_t i = 0; i < num_quads; i++) {
    // Create the face corners
    triangles[6 * i]     = 4 * i;
    triangles[6 * i + 1] = 4 * i + 2;
    triangles[6 * i + 2] = 4 * i + 3;
    triangles[6 * i + 3] = 4 * i + 3;
    triangles[6 * i + 4] = 4 * i + 1;
    triangles[6 * i + 5] = 4 * i;
  }

  // Build the AABB tree
  const Tree tree(Triangle_iterator(triangles),
                  Triangle_iterator(triangles + 6 * num_quads));
  tree.accelerate_distance_queries();

  // -------------------------------
  // Find visible points
  // -------------------------------
  // NOTE: Can't be parallelized currently
  // Now go through each described point and check if the point "is
  // visible" to the spherical camera center
  bool valid_kp[num_kp];
  size_t num_valid_kp = 0;
  for (size_t i = 0; i < num_kp; i++) {
    // Convert the spherical coordinate to XYZ coordinates
    const float x = kp_3d_ptr[3 * i];
    const float y = kp_3d_ptr[3 * i + 1];
    const float z = kp_3d_ptr[3 * i + 2];

    // Create a ray to query
    const auto pt = K::Point_3(x, y, z);
    const K::Ray_3 ray_query(CGAL::ORIGIN, pt);

    // Compute the ray-face intersection
    // Note that this function returns a boost::optional type which is why
    // there's a trailing `get()`
    const auto intersection_query = tree.first_intersection(ray_query);
    const auto intersection       = intersection_query.get();

    // The second part of the pair is the reference point idx of the
    // intersected face. <face_pt_idx> / 4 gives the "quad" index because
    // the first face of each quad is referenced by its point ID and each quad
    // has 4 points
    const size_t face_pt_idx = boost::get<size_t>(*intersection.second);
    const size_t quad_idx    = face_pt_idx / 4;

    // If the first intersected quad is the one on which the keypoint falls,
    // then keypoint is valid
    if (static_cast<int64_t>(quad_idx) == kp_quad_ptr[i]) {
      valid_kp[i] = true;
      num_valid_kp++;
    } else {
      valid_kp[i] = false;
    }
  }

  // Create output tensors
  torch::Tensor valid_kp_3d =
      torch::zeros({static_cast<int64_t>(num_valid_kp), 3}, torch::kFloat);
  torch::Tensor valid_kp_desc =
      torch::zeros({static_cast<int64_t>(num_valid_kp), 128}, torch::kFloat);
  torch::Tensor valid_kp_scale =
      torch::zeros({static_cast<int64_t>(num_valid_kp)}, torch::kFloat);
  torch::Tensor valid_kp_orient =
      torch::zeros({static_cast<int64_t>(num_valid_kp)}, torch::kFloat);
  auto valid_kp_3d_ptr   = valid_kp_3d.data<float>();
  auto valid_kp_desc_ptr = valid_kp_desc.data<float>();

  // Copy the data into the output tensors
  size_t cur_idx = 0;
  for (size_t i = 0; i < num_kp; i++) {
    if (valid_kp[i]) {
      std::copy(kp_desc_ptr + 128 * i, kp_desc_ptr + 128 * (i + 1),
                valid_kp_desc_ptr + 128 * cur_idx);
      std::copy(kp_3d_ptr + 3 * i, kp_3d_ptr + 3 * (i + 1),
                valid_kp_3d_ptr + 3 * cur_idx);
      valid_kp_scale[cur_idx]  = kp_scale_ptr[i];
      valid_kp_orient[cur_idx] = kp_orient_ptr[i];
      cur_idx++;
    }
  }

  return {valid_kp_3d, valid_kp_desc, valid_kp_scale, valid_kp_orient};
}

}  // namespace mesh
}  // namespace tangent_images