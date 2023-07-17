from tempfile import mkstemp
from shutil import move
import torch
import numpy as np
import os
from models.layers.vertex_mesh_union import VertexMeshUnion
from models.layers.vertex_mesh_prepare import fill_mesh

class VertexMesh(Mesh):
    def __init__(self, file=None, opt=None, hold_history=False, export_folder=''):
        super().__init__(file, opt, hold_history, export_folder)

    def extract_features(self):
        return self.features, self.vs

    def merge_vertices(self, edge_id):
        super().merge_vertices(edge_id)
        v_a = self.vs[self.edges[edge_id][0]]
        v_b = self.vs[self.edges[edge_id][1]]
        # Update vertex position
        self.vs[self.edges[edge_id][0]] = (v_a + v_b) / 2
        self.vs[self.edges[edge_id][1]] = np.nan

    def remove_vertex(self, v):
        super().remove_vertex(v)
        self.vs[v] = np.nan

    def clean(self, edges_mask, groups):
        super().clean(edges_mask, groups)
        self.vs = self.vs[self.v_mask]

    def export(self, file=None, vcolor=None):
        super().export(file, vcolor)

    def export_segments(self, segments):
        if not self.export_folder:
            return
        cur_segments = segments
        for i in range(self.pool_count + 1):
            filename, file_extension = os.path.splitext(self.filename)
            file = '%s/%s_%d%s' % (self.export_folder, filename, i, file_extension)
            fh, abs_path = mkstemp()
            edge_key = 0
            with os.fdopen(fh, 'w') as new_file:
                with open(file) as old_file:
                    for line in old_file:
                        if line[0] == 'e':
                            new_file.write('%s %d' % (line.strip(), cur_segments[edge_key]))
                            if edge_key < len(cur_segments):
                                edge_key += 1
                                new_file.write('\n')
                        else:
                            new_file.write(line)
            os.remove(file)
            move(abs_path, file)
            if i < len(self.history_data['edges_mask']):
                cur_segments = segments[:len(self.history_data['edges_mask'][i])]
                cur_segments = cur_segments[self.history_data['edges_mask'][i]]

    def __get_cycle(self, gemm, edge_id):
        cycles = []
        for j in range(2):
            next_side = start_point = j * 2
            next_key = edge_id
            if gemm[edge_id, start_point] == -1:
                continue
            cycles.append([])
            for i in range(3):
                tmp_next_key = gemm[next_key, next_side]
                tmp_next_side = self.sides[next_key, next_side]
                tmp_next_side = tmp_next_side + 1 - 2 * (tmp_next_side % 2)
                gemm[next_key, next_side] = -1
                gemm[next_key, next_side + 1 - 2 * (next_side % 2)] = -1
                next_key = tmp_next_key
                next_side = tmp_next_side
                cycles[-1].append(next_key)
        return cycles

    def __cycle_to_face(self, cycle, v_indices):
        face = []
        for i in range(3):
            v = list(set(self.edges[cycle[i]]) & set(self.edges[cycle[(i + 1) % 3]]))[0]
            face.append(v_indices[v])
        return face

    def init_history(self):
        self.history_data = {
            'groups': [],
            'gemm_edges': [self.gemm_edges.copy()],
            'occurrences': [],
            'old2current': np.arange(self.vs.shape[0], dtype=np.int32),
            'current2old': np.arange(self.vs.shape[0], dtype=np.int32),
            'edges_mask': [torch.ones(self.vs.shape[0], dtype=torch.bool)],
            'edges_count': [self.vs.shape[0]],
        }
        if self.export_folder:
            self.history_data['collapses'] = VertexMeshUnion(self.vs.shape[0])

    def union_groups(self, source, target):
        if self.export_folder and self.history_data:
            self.history_data['collapses'].union(self.history_data['current2old'][source],
                                                 self.history_data['current2old'][target])

    def remove_group(self, index):
        if self.history_data is not None:
            self.history_data['edges_mask'][-1][self.history_data['current2old'][index]] = 0
            self.history_data['old2current'][self.history_data['current2old'][index]] = -1
            if self.export_folder:
                self.history_data['collapses'].remove_group(self.history_data['current2old'][index])

    def get_groups(self):
        return self.history_data['groups'].pop()

    def get_occurrences(self):
        return self.history_data['occurrences'].pop()

    def __clean_history(self, groups, pool_mask):
        if self.history_data is not None:
            mask = self.history_data['old2current'] != -1
            self.history_data['old2current'][mask] = np.arange(self.vs.shape[0], dtype=np.int32)
            self.history_data['current2old'][0: self.vs.shape[0]] = np.ma.where(mask)[0]
            if self.export_folder != '':
                self.history_data['edges_mask'].append(self.history_data['edges_mask'][-1].clone())
            self.history_data['occurrences'].append(groups.get_occurrences())
            self.history_data['groups'].append(groups.get_groups(pool_mask))
            self.history_data['gemm_edges'].append(self.gemm_edges.copy())
            self.history_data['edges_count'].append(self.vs.shape[0])

    def unroll_gemm(self):
        self.history_data['gemm_edges'].pop()
        self.gemm_edges = self.history_data['gemm_edges'][-1]
        self.history_data['edges_count'].pop()
        self.edges_count = self.history_data['edges_count'][-1]