import numpy as np
import os
import ntpath


def fill_mesh(self, file: str, opt):
        load_path = self.get_mesh_path(file, opt.num_aug)
        if os.path.exists(load_path):
            mesh_data = np.load(load_path, encoding='latin1', allow_pickle=True)
        else:
            mesh_data = self.from_scratch(file, opt)
            np.savez_compressed(load_path, vs=mesh_data.vs, ve=mesh_data.ve,
                                v_mask=mesh_data.v_mask, filename=mesh_data.filename,
                                edge_lengths=mesh_data.edge_lengths,
                                edge_areas=mesh_data.edge_areas, features=mesh_data.features)
        self.vs = mesh_data['vs']
        self.ve = mesh_data['ve']
        self.v_mask = mesh_data['v_mask']
        self.filename = str(mesh_data['filename'])
        self.edge_lengths = mesh_data['edge_lengths']
        self.edge_areas = mesh_data['edge_areas']
        self.features = mesh_data['features']
        self.vertices_count = len(self.vs)

    
def get_mesh_path(file: str, num_aug: int):
        filename, _ = os.path.splitext(file)
        dir_name = os.path.dirname(filename)
        prefix = os.path.basename(filename)
        load_dir = os.path.join(dir_name, 'cache')
        load_file = os.path.join(load_dir, '%s_%03d.npz' % (prefix, np.random.randint(0, num_aug)))
        if not os.path.isdir(load_dir):
            os.makedirs(load_dir, exist_ok=True)
        return load_file

    
def from_scratch(file, opt):
        
        class MeshPrep:
            def __init__(self):
                self.vs = None
                self.ve = None
                self.v_mask = None
                self.filename = None
                self.edge_lengths = None
                self.edge_areas = None
                self.features = None
                self.vertices_count = None

        mesh_data = MeshPrep()
        mesh_data.vs, faces = fill_from_file(mesh_data, file)
        mesh_data.v_mask = np.ones(len(mesh_data.vs), dtype=bool)
        faces, face_areas = remove_non_manifolds(mesh_data, faces)
        if opt.num_aug > 1:
            faces = augmentation(mesh_data, opt, faces)
        compute_edge_lengths(mesh_data, faces)
        mesh_data.features = extract_features(mesh_data)
        return mesh_data


def fill_from_file(mesh, file):
        mesh.filename = ntpath.split(file)[1]
        mesh.fullfilename = file
        vs, faces = [], []
        f = open(file)
        for line in f:
            line = line.strip()
            splitted_line = line.split()
            if not splitted_line:
                continue
            elif splitted_line[0] == 'v':
                vs.append([float(v) for v in splitted_line[1:4]])
            elif splitted_line[0] == 'f':
                face_vertex_ids = [int(c.split('/')[0]) for c in splitted_line[1:]]
                assert len(face_vertex_ids) == 3
                face_vertex_ids = [(ind - 1) if (ind >= 0) else (len(vs) + ind)
                                   for ind in face_vertex_ids]
                faces.append(face_vertex_ids)
        f.close()
        vs = np.asarray(vs)
        faces = np.asarray(faces, dtype=int)
        assert np.logical_and(faces >= 0, faces < len(vs)).all()
        return vs, faces


def remove_non_manifolds(mesh, faces):
        mesh.ve = [[] for _ in mesh.vs]
        edges_set = set()
        mask = np.ones(len(faces), dtype=bool)
        _, face_areas = compute_face_normals_and_areas(mesh, faces)
        for face_id, face in enumerate(faces):
            if face_areas[face_id] == 0:
                mask[face_id] = False
                continue
            faces_edges = []
            is_manifold = False
            for i in range(3):
                cur_edge = (face[i], face[(i + 1) % 3])
                if cur_edge in edges_set:
                    is_manifold = True
                    break
                else:
                    faces_edges.append(cur_edge)
            if is_manifold:
                mask[face_id] = False
            else:
                for idx, edge in enumerate(faces_edges):
                    edges_set.add(edge)
        return faces[mask], face_areas[mask]

def compute_face_normals_and_areas(mesh, faces):
        face_normals = np.cross(mesh.vs[faces[:, 1]] - mesh.vs[faces[:, 0]],
                                mesh.vs[faces[:, 2]] - mesh.vs[faces[:, 1]])
        face_areas = np.sqrt((face_normals ** 2).sum(axis=1))
        face_normals /= face_areas[:, np.newaxis]
        assert (not np.any(face_areas[:, np.newaxis] == 0)), 'has zero area face: %s' % mesh.filename
        face_areas *= 0.5
        return face_normals, face_areas

def augmentation(mesh, opt, faces=None):
        if hasattr(opt, 'scale_verts') and opt.scale_verts:
            scale_verts(mesh)
        if hasattr(opt, 'flip_edges') and opt.flip_edges:
            faces = flip_edges(mesh, opt.flip_edges, faces)
        return faces

def compute_edge_lengths(mesh, faces):
        edge_lengths = []
        for face in faces:
            for i in range(3):
                v1 = mesh.vs[face[i]]
                v2 = mesh.vs[face[(i + 1) % 3]]
                edge_lengths.append(np.linalg.norm(v2 - v1))
        mesh.edge_lengths = np.array(edge_lengths, dtype=np.float32)

def extract_features(mesh):
        features = []
        edge_points = get_edge_points(mesh)
        compute_edge_lengths(mesh, edge_points)
        with np.errstate(divide='raise'):
            try:
                for extractor in [dihedral_angle, symmetric_opposite_angles, symmetric_ratios]:
                    feature = extractor(mesh, edge_points)
                    features.append(feature)
                return np.concatenate(features, axis=0)
            except Exception as e:
                print(e)
                raise ValueError(mesh.filename, 'bad features')

def dihedral_angle(mesh, edge_points):
        normals_a = get_normals(mesh, edge_points, 0)
        normals_b = get_normals(mesh, edge_points, 3)
        dot = np.sum(normals_a * normals_b, axis=1).clip(-1, 1)
        angles = np.expand_dims(np.pi - np.arccos(dot), axis=0)
        return angles

def symmetric_opposite_angles(mesh, edge_points):
        angles_a = get_opposite_angles(mesh, edge_points, 0)
        angles_b = get_opposite_angles(mesh, edge_points, 3)
        angles = np.concatenate((np.expand_dims(angles_a, 0), np.expand_dims(angles_b, 0)), axis=0)
        angles = np.sort(angles, axis=0)
        return angles

def symmetric_ratios(mesh, edge_points):
        ratios_a = get_ratios(mesh, edge_points, 0)
        ratios_b = get_ratios(mesh, edge_points, 3)
        ratios = np.concatenate((np.expand_dims(ratios_a, 0), np.expand_dims(ratios_b, 0)), axis=0)
        return np.sort(ratios, axis=0)

def get_edge_points(mesh):
        edge_points = np.zeros([mesh.edges_count, 4], dtype=np.int32)
        for edge_id, edge in enumerate(mesh.edges):
            edge_points[edge_id] = get_side_points(mesh, edge_id)
        return edge_points

def get_side_points(mesh, edge_id):
        edge_a = mesh.edges[edge_id]

        if mesh.gemm_edges[edge_id, 0] == -1:
            edge_b = mesh.edges[mesh.gemm_edges[edge_id, 2]]
            edge_c = mesh.edges[mesh.gemm_edges[edge_id, 3]]
        else:
            edge_b = mesh.edges[mesh.gemm_edges[edge_id, 0]]
            edge_c = mesh.edges[mesh.gemm_edges[edge_id, 1]]
        if mesh.gemm_edges[edge_id, 2] == -1:
            edge_d = mesh.edges[mesh.gemm_edges[edge_id, 0]]
            edge_e = mesh.edges[mesh.gemm_edges[edge_id, 1]]
        else:
            edge_d = mesh.edges[mesh.gemm_edges[edge_id, 2]]
            edge_e = mesh.edges[mesh.gemm_edges[edge_id, 3]]
        first_vertex = 0
        second_vertex = 0
        third_vertex = 0
        if edge_a[1] in edge_b:
            first_vertex = 1
        if edge_b[1] in edge_c:
            second_vertex = 1
        if edge_d[1] in edge_e:
            third_vertex = 1
        return [edge_a[first_vertex], edge_a[1 - first_vertex], edge_b[second_vertex], edge_d[third_vertex]]

def get_normals(mesh, edge_points, side):
        edge_a = mesh.vs[edge_points[:, side // 2 + 2]] - mesh.vs[edge_points[:, side // 2]]
        edge_b = mesh.vs[edge_points[:, 1 - side // 2]] - mesh.vs[edge_points[:, side // 2]]
        normals = np.cross(edge_a, edge_b)
        div = fixed_division(np.linalg.norm(normals, ord=2, axis=1), epsilon=0.1)
        normals /= div[:, np.newaxis]
        return normals

def get_opposite_angles(mesh, edge_points, side):
        edges_a = mesh.vs[edge_points[:, side // 2]] - mesh.vs[edge_points[:, side // 2 + 2]]
        edges_b = mesh.vs[edge_points[:, 1 - side // 2]] - mesh.vs[edge_points[:, side // 2 + 2]]

        edges_a /= fixed_division(np.linalg.norm(edges_a, ord=2, axis=1), epsilon=0.1)[:, np.newaxis]
        edges_b /= fixed_division(np.linalg.norm(edges_b, ord=2, axis=1), epsilon=0.1)[:, np.newaxis]
        dot = np.sum(edges_a * edges_b, axis=1).clip(-1, 1)
        return np.arccos(dot)

def get_ratios(mesh, edge_points, side):
        edges_lengths = np.linalg.norm(mesh.vs[edge_points[:, side // 2]] - mesh.vs[edge_points[:, 1 - side // 2]],
                                       ord=2, axis=1)
        point_o = mesh.vs[edge_points[:, side // 2 + 2]]
        point_a = mesh.vs[edge_points[:, side // 2]]
        point_b = mesh.vs[edge_points[:, 1 - side // 2]]
        line_ab = point_b - point_a
        projection_length = np.sum(line_ab * (point_o - point_a), axis=1) / fixed_division(
            np.linalg.norm(line_ab, ord=2, axis=1), epsilon=0.1)
        closest_point = point_a + (projection_length / edges_lengths)[:, np.newaxis] * line_ab
        d = np.linalg.norm(point_o - closest_point, ord=2, axis=1)
        return d / edges_lengths

def fixed_division(to_div, epsilon):
        if epsilon == 0:
            to_div[to_div == 0] = 0.1
        else:
            to_div += epsilon
        return to_div