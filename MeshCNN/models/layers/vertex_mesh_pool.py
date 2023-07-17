import torch
import torch.nn as nn
from threading import Thread
from models.layers.vertex_mesh_union import VertexMeshUnion
import numpy as np
from heapq import heappop, heapify


class VertexMeshPool(nn.Module):
    def __init__(self, target, multi_thread=False):
        super(VertexMeshPool, self).__init__()
        self.__out_target = target
        self.__multi_thread = multi_thread
        self.__fe = None
        self.__updated_fe = None
        self.__meshes = None
        self.__merge_vertex = [-1]

    def __call__(self, fe, meshes):
        return self.forward(fe, meshes)

    def forward(self, fe, meshes):
        self.__updated_fe = [[] for _ in range(len(meshes))]
        pool_threads = []
        self.__fe = fe
        self.__meshes = meshes
        # iterate over batch
        for mesh_index in range(len(meshes)):
            if self.__multi_thread:
                pool_threads.append(Thread(target=self.__pool_main, args=(mesh_index,)))
                pool_threads[-1].start()
            else:
                self.__pool_main(mesh_index)
        if self.__multi_thread:
            for mesh_index in range(len(meshes)):
                pool_threads[mesh_index].join()
        out_features = torch.cat(self.__updated_fe).view(len(meshes), -1, self.__out_target)
        return out_features

    def __pool_main(self, mesh_index):
        mesh = self.__meshes[mesh_index]
        queue = self.__build_queue(self.__fe[mesh_index, :, :mesh.vertices_count], mesh.vertices_count)
        mask = np.ones(mesh.vertices_count, dtype=np.bool)
        vertex_groups = VertexMeshUnion(mesh.vertices_count, self.__fe.device)
        while mesh.vertices_count > self.__out_target:
            value, vertex_id = heappop(queue)
            vertex_id = int(vertex_id)
            if mask[vertex_id]:
                self.__pool_vertex(mesh, vertex_id, mask, vertex_groups)
        mesh.clean(mask, vertex_groups)
        fe = vertex_groups.rebuild_features(self.__fe[mesh_index], mask, self.__out_target)
        self.__updated_fe[mesh_index] = fe

    def __pool_vertex(self, mesh, vertex_id, mask, vertex_groups):
        if self.has_boundaries(mesh, vertex_id):
            return False
        elif self.__clean_ring(mesh, vertex_id, mask, vertex_groups):
            self.__merge_vertex[0] = self.__pool_ring(mesh, vertex_id, mask, vertex_groups)
            mesh.merge_vertices(vertex_id)
            mask[vertex_id] = False
            self.__remove_group(mesh, vertex_groups, vertex_id)
            mesh.vertices_count -= 1
            return True
        else:
            return False

    def __clean_ring(self, mesh, vertex_id, mask, vertex_groups):
        if mesh.vertices_count <= self.__out_target:
            return False
        invalid_vertices = self.__get_invalids(mesh, vertex_id, vertex_groups)
        while len(invalid_vertices) != 0 and mesh.vertices_count > self.__out_target:
            self.__remove_triangle(mesh, mask, vertex_groups, invalid_vertices)
            if mesh.vertices_count <= self.__out_target:
                return False
            if self.has_boundaries(mesh, vertex_id):
                return False
            invalid_vertices = self.__get_invalids(mesh, vertex_id, vertex_groups)
        return True

    @staticmethod
    def has_boundaries(mesh, vertex_id):
        for vertex in mesh.ve[vertex_id]:
            if vertex == -1 or -1 in mesh.ve[vertex]:
                return True
        return False

    def __get_invalids(self, mesh, vertex_id, vertex_groups):
        invalid_vertices = []
        for neighbor_id in mesh.ve[vertex_id]:
            if neighbor_id != -1 and neighbor_id != vertex_id:
                if not self.__validate_ring(mesh, vertex_id, neighbor_id):
                    invalid_vertices.append(neighbor_id)
        return invalid_vertices

    @staticmethod
    def __validate_ring(mesh, vertex_id, neighbor_id):
        neighbors_vertex_ids = [mesh.ve[neighbor_id][i] for i in range(len(mesh.ve[neighbor_id]))]
        return vertex_id in neighbors_vertex_ids

    def __remove_triangle(self, mesh, mask, vertex_groups, invalid_vertices):
        vertices = set([invalid_vertices[0]])
        for vertex_id in invalid_vertices:
            vertices.add(vertex_id)
            mask[vertex_id] = False
            self.__remove_group(mesh, vertex_groups, vertex_id)
        mesh.vertices_count -= len(invalid_vertices)
        vertices = list(vertices)
        mesh.remove_vertex(vertices[0])

    def __pool_ring(self, mesh, vertex_id, mask, vertex_groups):
        neighbor_ids = mesh.ve[vertex_id]
        valid_neighbors = [neighbor_id for neighbor_id in neighbor_ids if neighbor_id != -1 and mask[neighbor_id]]
        if len(valid_neighbors) == 0:
            return -1
        else:
            return valid_neighbors[0]

    @staticmethod
    def __build_queue(features, vertices_count):
        # delete vertices with smallest norm
        squared_magnitude = torch.sum(features * features, 0)
        if squared_magnitude.shape[-1] != 1:
            squared_magnitude = squared_magnitude.unsqueeze(-1)
        vertex_ids = torch.arange(vertices_count, device=squared_magnitude.device, dtype=torch.float32).unsqueeze(-1)
        heap = torch.cat((squared_magnitude, vertex_ids), dim=-1).tolist()
        heapify(heap)
        return heap

    @staticmethod
    def __union_groups(mesh, vertex_groups, source, target):
        vertex_groups.union(source, target)
        mesh.union_groups(source, target)

    @staticmethod
    def __remove_group(mesh, vertex_groups, index):
        vertex_groups.remove_group(index)
        mesh.remove_group(index)