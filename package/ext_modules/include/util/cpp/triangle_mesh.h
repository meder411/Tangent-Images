// Currently set up for inheritance of for Mesh --> TriangleMesh --> Icosphere
//                                             |--> QuadMesh
// But issues with PyBind11 mean that I will add this to the TODO until after
// CVPR

#ifndef CGAL_MESH_H_
#define CGAL_MESH_H_

#include <CGAL/Polygon_mesh_processing/measure.h>
#include <CGAL/Simple_cartesian.h>
#include <CGAL/Surface_mesh.h>

#include <boost/iterator/iterator_adaptor.hpp>

#include <pybind11/operators.h>

#include <torch/extension.h>

#include <algorithm>
#include <list>
#include <map>
#include <vector>

typedef CGAL::Simple_cartesian<double> K;
typedef CGAL::Surface_mesh<K::Point_3> SurfaceMesh;
typedef SurfaceMesh::Vertex_index VertexDescriptor;
typedef SurfaceMesh::Face_index FaceDescriptor;

namespace tangent_images {
namespace mesh {

class TriangleMesh {
 private:
  SurfaceMesh _mesh;  // The CGAL SurfaceMesh object

  // Construction functions
  const VertexDescriptor _AddVertex(const K::Point_3 &pt);
  const FaceDescriptor _AddFace(const VertexDescriptor &v0,
                                const VertexDescriptor &v1,
                                const VertexDescriptor &v2);
  void _BuildMesh(const float *pts, const size_t num_pts, const int64_t *faces,
                  const size_t num_faces);

  // Accessor functions that operate on individual faces
  const K::Point_3 _GetFaceBarycenter(const FaceDescriptor &fd) const;
  const std::vector<int64_t> _GetFacesAdjacentToFace(
      const FaceDescriptor &fd) const;
  const std::vector<int64_t> _GetVerticesAdjacentToFace(
      const FaceDescriptor &fd) const;

  // Accessor functions that operate on individual vertices
  const std::vector<int64_t> _GetFacesAdjacentToVertex(
      const VertexDescriptor &vd) const;
  const std::vector<int64_t> _GetVerticesAdjacentToVertex(
      const VertexDescriptor &vd) const;

  // Distance computations per face
  const std::vector<float> _GetBarycentricWeights(
      const K::Point_3 &pt, const FaceDescriptor &fd) const;

  // Manipulations on vertices
  void _Add(const float val);
  void _Subtract(const float val);
  void _Scale(const float val);

 public:
  typedef std::unique_ptr<TriangleMesh> unique_ptr;

  // Constructor
  TriangleMesh();
  TriangleMesh(torch::Tensor pts, torch::Tensor faces);
  TriangleMesh(const std::vector<float> &pts,
               const std::vector<int64_t> &faces);

  // Output streem override
  friend std::ostream &operator<<(std::ostream &out, const TriangleMesh &data);

  // Operator overloads
  TriangleMesh &operator+=(const float rhs) {
    this->_Add(rhs);
    return *this;
  }
  TriangleMesh &operator-=(const float rhs) {
    this->_Add(-rhs);
    return *this;
  }
  TriangleMesh &operator*=(const float rhs) {
    this->_Scale(rhs);
    return *this;
  }
  TriangleMesh &operator/=(const float rhs) {
    this->_Scale(1.0 / rhs);
    return *this;
  }
  friend TriangleMesh operator+(TriangleMesh lhs, const float rhs) {
    lhs._Add(rhs);
    return lhs;
  }
  friend TriangleMesh operator-(TriangleMesh lhs, const float rhs) {
    lhs._Add(-rhs);
    return lhs;
  }
  friend TriangleMesh operator*(TriangleMesh lhs, const float rhs) {
    lhs._Scale(rhs);
    return lhs;
  }
  friend TriangleMesh operator/(TriangleMesh lhs, const float rhs) {
    lhs._Scale(1.0 / rhs);
    return lhs;
  }

  // Simple inlined functions
  inline SurfaceMesh &GetMesh();
  inline const size_t NumVertices() const;
  inline const size_t NumFaces() const;
  inline const size_t VerticesPerFace() const;
  inline const float SurfaceArea() const;

  // Static functions
  // Computes the norm of a point
  static const float PointNorm(const K::Point_3 &pt);
  static const float VectorNorm(const K::Vector_3 &pt);
  static const std::vector<float> ComputeBarycentricCoordinates(
      const K::Point_3 &pt, const K::Point_3 &A, const K::Point_3 &B,
      const K::Point_3 &C);

  const std::string ToString() const;

  // Calls Loop subdivision
  void LoopSubdivide(const size_t order);

  void CatmullClarkSubdivide(const size_t order);

  void MidpointSubdivide(const size_t order);

  void NormalizePoints();

  // Returns F x 3 x 3, F sets of 3 rows of points (each row is a point)
  const torch::Tensor GetAllFaceVertexCoordinates() const;

  // Returns F x 3, F rows of 3 indices
  const torch::Tensor GetAllFaceVertexIndices() const;

  // Returns F x 3
  const torch::Tensor GetAdjacentFaceIndicesToFaces() const;

  // Returns a dictionary with V keys, each mapping to a 1-D Tensor containing
  // the adjacent vertex indices
  const std::map<int64_t, torch::Tensor> GetAdjacentVertexIndicesToVertices()
      const;

  // Returns a dictionary with V keys, each mapping to a 1-D Tensor containing
  // the adjacent face indices
  const std::map<int64_t, torch::Tensor> GetAdjacentFaceIndicesToVertices()
      const;

  // Returns V x 3
  const torch::Tensor GetVertices() const;

  // Returns {V x 3 vertex normals, F x 3 face normals}
  const std::vector<torch::Tensor> ComputeNormals();

  // Returns the radius of a spheroid mesh as the average of point norms
  const float SpheroidRadius() const;

  // Returns the average distance between vertices
  const float GetVertexResolution() const;

  // Returns the average angle between vertices
  const float GetAngularResolution() const;

  // Returns F x 3, F rows of triangle barycenters
  const torch::Tensor GetFaceBarycenters() const;

  const torch::Tensor GetFacesAdjacentToFace(const size_t face_idx) const;
  const torch::Tensor GetVerticesAdjacentToFace(const size_t face_idx) const;
  const torch::Tensor GetFacesAdjacentToVertex(const size_t vertex_idx) const;
  const torch::Tensor GetVerticesAdjacentToVertex(
      const size_t vertex_idx) const;

  // Returns OH x OW x Kh*Kw
  static const std::vector<torch::Tensor> GetIcosphereConvolutionOperator(
      torch::Tensor samples, const TriangleMesh &icosphere,
      const bool keepdim = false, const bool nearest = false);

  static const torch::Tensor DistortImageGrid(
      TriangleMesh &image_grid, const std::vector<float> &radial,
      const std::vector<float> tangential);

  static const torch::Tensor GetFaceTuples(const size_t order);
};

// Inline definitions
SurfaceMesh &TriangleMesh::GetMesh() {
  return this->_mesh;
}
const size_t TriangleMesh::NumVertices() const {
  return this->_mesh.number_of_vertices();
}
const size_t TriangleMesh::NumFaces() const {
  return this->_mesh.number_of_faces();
}
const float TriangleMesh::SurfaceArea() const {
  return CGAL::Polygon_mesh_processing::area(this->_mesh);
}

// ----------------------------------------------------------------------------
// Non-class functions
// ----------------------------------------------------------------------------
inline const TriangleMesh GenerateIcosphere(const size_t order) {
  // Generates an icosahedron such that the 0-th point is (0,1,0)

  std::vector<float> pts = std::vector<float>{
      0.0,          1.0,          0.0,          0.894427191,  0.447213595,
      0.0,          -0.894427191, -0.447213595, 0.0,          0.0,
      -1.0,         0.0,          -0.276393202, -0.447213595, 0.850650808,
      0.276393202,  0.447213595,  0.850650808,  -0.276393202, -0.447213595,
      -0.850650808, 0.276393202,  0.447213595,  -0.850650808, 0.723606798,
      -0.447213595, -0.525731112, 0.723606798,  -0.447213595, 0.525731112,
      -0.723606798, 0.447213595,  -0.525731112, -0.723606798, 0.447213595,
      0.525731112};
  std::vector<int64_t> face_indices = std::vector<int64_t>{
      0, 11, 5,  0, 5,  1, 0, 1, 7, 0, 7,  10, 0, 10, 11, 1, 5, 9, 5, 11,
      4, 11, 10, 2, 10, 7, 6, 7, 1, 8, 3,  9,  4, 3,  4,  2, 3, 2, 6, 3,
      6, 8,  3,  8, 9,  5, 4, 9, 2, 4, 11, 6,  2, 10, 8,  6, 7, 9, 8, 1};

  TriangleMesh icosphere(pts, face_indices);
  icosphere.LoopSubdivide(order);
  return icosphere;
}

inline const TriangleMesh GenerateOctosphere(const size_t order) {
  std::vector<float> pts =
      std::vector<float>{1.0,  0.0, 0.0, 0.0, 1.0,  0.0, 0.0, 0.0, 1.0,
                         -1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0};
  std::vector<int64_t> face_indices = std::vector<int64_t>{
      0, 1, 2, 0, 2, 4, 0, 4, 5, 0, 5, 1, 3, 1, 5, 3, 5, 4, 3, 4, 2, 3, 2, 1};

  TriangleMesh icosphere(pts, face_indices);
  icosphere.LoopSubdivide(order);
  return icosphere;
}

inline const TriangleMesh GenerateCube(const size_t order) {
  std::vector<float> pts = std::vector<float>{
      -0.5, -0.5, -0.5, 0.5, -0.5, -0.5, -0.5, 0.5, -0.5, 0.5, 0.5, -0.5,
      -0.5, -0.5, 0.5,  0.5, -0.5, 0.5,  -0.5, 0.5, 0.5,  0.5, 0.5, 0.5};
  std::vector<int64_t> face_indices = std::vector<int64_t>{
      2, 1, 0, 1, 2, 3, 4, 2, 0, 2, 4, 6, 1, 4, 0, 4, 1, 5,
      6, 5, 7, 5, 6, 4, 3, 6, 7, 6, 3, 2, 5, 3, 7, 3, 5, 1};

  TriangleMesh icosphere(pts, face_indices);
  icosphere.CatmullClarkSubdivide(order);
  return icosphere;
}

inline const TriangleMesh GenerateIcosahedralNet(const size_t order) {
  const float h = sqrt(3.0) / 2.0;

  std::vector<float> pts = std::vector<float>{
      0.5,    h,   0.0, 3.0,    0.0, 0.0,    0.5, -h,  0.0,    0.0, -2 * h,
      0.0,    2.0, 0.0, 0.0,    1.5, -h,     0.0, 4.0, 0.0,    0.0, -0.5,
      -h,     0.0, 1.0, 0.0,    0.0, 0.0,    0.0, 0.0, 2.5,    -h,  0.0,
      3.5,    -h,  0.0, 1.5,    h,   0.0,    2.5, h,   0.0,    3.5, h,
      0.0,    4.5, h,   0.0,    1.0, -2 * h, 0.0, 2.0, -2 * h, 0.0, 3.0,
      -2 * h, 0.0, 4.0, -2 * h, 0.0, 4.5,    -h,  0.0, 5.0,    0.0, 0.0};
  std::vector<int64_t> face_indices = std::vector<int64_t>{
      9, 8, 0,  12, 8,  4, 1,  13, 4, 14, 1,  6,  15, 6,  21,
      2, 9, 7,  8,  9,  2, 2,  5,  8, 5,  4,  8,  4,  5,  10,
      1, 4, 10, 10, 11, 1, 1,  11, 6, 20, 6,  11, 6,  20, 21,
      3, 2, 7,  2,  16, 5, 17, 10, 5, 11, 10, 18, 19, 20, 11};

  TriangleMesh net(pts, face_indices);
  // net.Subdivide(order);
  return net;
}

inline const TriangleMesh GenerateImageGrid(size_t height, size_t width,
                                            const bool half_step = false) {
  std::vector<float> pts;
  std::vector<int64_t> face_indices;

  // Adjust the size of the grid to allow pixels to be the center of a quad
  // face (although represented a triangle mesh)
  if (half_step) {
    height += 1;
    width += 1;
  }

  // Normalize point spacing
  const float normalizer =
      2.0 / static_cast<float>(std::max(width, height) - 1);

  // Create points vector
  for (size_t y = 0; y < height; y++) {
    for (size_t x = 0; x < width; x++) {
      pts.push_back(static_cast<float>(x - ((width - 1) / 2.0)) * normalizer);
      pts.push_back(static_cast<float>(y - ((height - 1) / 2.0)) * normalizer);
      pts.push_back(0.0);
    }
  }

  // Create face indices vector where each pixel is represented as a face if
  // half_step is true and a vertex otherwise
  //
  // (x, y)  (x+1, y)
  //  * ---- *
  //  | \   |
  //  |  \  |
  //  |   \ |
  //  * ---- *
  // (x, y+1)  (x+1, y+1)
  for (size_t y = 0; y < height - 1; y++) {
    for (size_t x = 0; x < width - 1; x++) {
      const int64_t vtl_idx = y * width + x;
      const int64_t vtr_idx = y * width + x + 1;
      const int64_t vbl_idx = (y + 1) * width + x;
      const int64_t vbr_idx = (y + 1) * width + x + 1;

      // Lower-left triangle indices
      face_indices.push_back(vtl_idx);
      face_indices.push_back(vbl_idx);
      face_indices.push_back(vbr_idx);

      // Upper-right triangle indices
      face_indices.push_back(vbr_idx);
      face_indices.push_back(vtr_idx);
      face_indices.push_back(vtl_idx);
    }
  }

  TriangleMesh mesh(pts, face_indices);
  return mesh;
}

std::vector<torch::Tensor> FindTangentPlaneIntersections(
    torch::Tensor plane_corners, torch::Tensor rays);

std::vector<torch::Tensor> FindVisibleKeypoints(torch::Tensor kp_3d,
                                                torch::Tensor kp_quad,
                                                torch::Tensor kp_desc,
                                                torch::Tensor kp_scale,
                                                torch::Tensor kp_orient,
                                                torch::Tensor plane_corners);

class MidpointSubdivisionMask {
  typedef typename boost::graph_traits<SurfaceMesh>::vertex_descriptor
      vertex_descriptor;
  typedef typename boost::graph_traits<SurfaceMesh>::halfedge_descriptor
      halfedge_descriptor;
  typedef typename boost::property_map<SurfaceMesh, CGAL::vertex_point_t>::type
      Vertex_pmap;
  typedef typename boost::property_traits<Vertex_pmap>::value_type Point;
  typedef typename boost::property_traits<Vertex_pmap>::reference Point_ref;
  SurfaceMesh &pmesh;
  Vertex_pmap vpm;

 public:
  MidpointSubdivisionMask(SurfaceMesh &pmesh)
      : pmesh(pmesh), vpm(get(CGAL::vertex_point, pmesh)) {}
  void edge_node(halfedge_descriptor hd, Point &pt) {
    Point_ref p1 = get(vpm, target(hd, pmesh));
    Point_ref p2 = get(vpm, target(opposite(hd, pmesh), pmesh));
    pt = Point((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2, (p1[2] + p2[2]) / 2);
  }
  void vertex_node(vertex_descriptor vd, Point &pt) {
    Point_ref S = get(vpm, vd);
    pt          = Point(S[0], S[1], S[2]);
  }
  void border_node(halfedge_descriptor hd, Point &ept, Point &vpt) {
    this->edge_node(hd, ept);
    this->vertex_node(target(hd, pmesh), vpt);
  }
};

}  // namespace mesh
}  // namespace tangent_images

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // TriangleMesh class
  py::class_<tangent_images::mesh::TriangleMesh>(m, "TriangleMesh")
      .def(py::init<torch::Tensor, torch::Tensor>())
      .def("__str__", &tangent_images::mesh::TriangleMesh::ToString)
      .def("__repr__", &tangent_images::mesh::TriangleMesh::ToString)
      .def(py::self += float())
      .def(py::self -= float())
      .def(py::self *= float())
      .def(py::self /= float())
      .def(py::self + float())
      .def(py::self - float())
      .def(py::self * float())
      .def(py::self / float())
      .def("num_vertices", &tangent_images::mesh::TriangleMesh::NumVertices)
      .def("num_faces", &tangent_images::mesh::TriangleMesh::NumFaces)
      .def("get_vertices", &tangent_images::mesh::TriangleMesh::GetVertices)
      .def("surface_area", &tangent_images::mesh::TriangleMesh::SurfaceArea)
      .def("radius", &tangent_images::mesh::TriangleMesh::SpheroidRadius)
      .def("compute_normals",
           &tangent_images::mesh::TriangleMesh::ComputeNormals)
      .def("normalize_points",
           &tangent_images::mesh::TriangleMesh::NormalizePoints)
      .def("loop_subdivision",
           &tangent_images::mesh::TriangleMesh::LoopSubdivide)
      .def("midpoint_subdivision",
           &tangent_images::mesh::TriangleMesh::MidpointSubdivide)
      .def("catmull_clark_subdivision",
           &tangent_images::mesh::TriangleMesh::CatmullClarkSubdivide)
      .def("get_faces_adjacent_to_face",
           &tangent_images::mesh::TriangleMesh::GetFacesAdjacentToFace)
      .def("get_vertices_adjacent_to_face",
           &tangent_images::mesh::TriangleMesh::GetVerticesAdjacentToFace)
      .def("get_faces_adjacent_to_vertex",
           &tangent_images::mesh::TriangleMesh::GetFacesAdjacentToVertex)
      .def("get_vertices_adjacent_to_vertex",
           &tangent_images::mesh::TriangleMesh::GetVerticesAdjacentToVertex)
      .def("get_all_face_vertex_coords",
           &tangent_images::mesh::TriangleMesh::GetAllFaceVertexCoordinates)
      .def("get_all_face_vertex_indices",
           &tangent_images::mesh::TriangleMesh::GetAllFaceVertexIndices)
      .def("get_face_barycenters",
           &tangent_images::mesh::TriangleMesh::GetFaceBarycenters)
      .def("get_adjacent_face_indices_to_faces",
           &tangent_images::mesh::TriangleMesh::GetAdjacentFaceIndicesToFaces)
      .def("get_adjacent_face_indices_to_vertices",
           &tangent_images::mesh::TriangleMesh::
               GetAdjacentFaceIndicesToVertices)
      .def("get_adjacent_vertex_indices_to_vertices",
           &tangent_images::mesh::TriangleMesh::
               GetAdjacentVertexIndicesToVertices)
      .def("get_vertex_resolution",
           &tangent_images::mesh::TriangleMesh::GetVertexResolution)
      .def("get_angular_resolution",
           &tangent_images::mesh::TriangleMesh::GetAngularResolution);

  // Other functions
  m.def("generate_icosphere", &tangent_images::mesh::GenerateIcosphere)
      .def("generate_octosphere", &tangent_images::mesh::GenerateOctosphere)
      .def("generate_image_grid_mesh",
           &tangent_images::mesh::GenerateImageGrid)
      .def("generate_cube", &tangent_images::mesh::GenerateCube)
      .def("generate_icosahedral_net",
           &tangent_images::mesh::GenerateIcosahedralNet)
      .def(
          "get_icosphere_convolution_operator",
          &tangent_images::mesh::TriangleMesh::GetIcosphereConvolutionOperator)
      .def("get_face_tuples",
           &tangent_images::mesh::TriangleMesh::GetFaceTuples)
      .def("find_tangent_plane_intersections",
           &tangent_images::mesh::FindTangentPlaneIntersections)
      .def("find_visible_keypoints",
           &tangent_images::mesh::FindVisibleKeypoints)
      .def("distort_image_grid",
           &tangent_images::mesh::TriangleMesh::DistortImageGrid);
}

#endif