/*
 * cc.cpp
 *
 *  Created on: Aug 8, 2017
 *      Author: kai
 */

#include "../core/engine.hpp"
#include "../core/scatter.hpp"
#include "../core/gather.hpp"
#include "../core/global_info.hpp"

using namespace RStream;

static VertexId start_vertex_id = 0;
const static VertexId DEFAULT_MAX_DISTANCE = std::numeric_limits<VertexId>::max();
struct Update_SSSP : BaseUpdate {
    VertexId distance;

    Update_SSSP(VertexId _target, VertexId _component) : BaseUpdate(_target), distance(_component) {};

    Update_SSSP() : BaseUpdate(INVALID_VERTEX_ID), distance(DEFAULT_MAX_DISTANCE) {}

    std::string toString() {
        return "(" + std::to_string(target) + ", " + std::to_string(distance) + ")";
    }
}__attribute__((__packed__));

struct Vertex_SSSP : BaseVertex {
    VertexId distance;
}__attribute__((__packed__));

void init(char *data, VertexId id) {
    struct Vertex_SSSP *v = (struct Vertex_SSSP *) data;
    v->id = id;
    if(start_vertex_id == id) {
        v->distance = 0;
    } else {
        v->distance = DEFAULT_MAX_DISTANCE;
    }
}

Update_SSSP *generate_one_update(Edge *e, Vertex_SSSP *v) {
    Update_SSSP *update;
    if(v->distance == DEFAULT_MAX_DISTANCE) {
        update = new Update_SSSP;
    } else {
        VertexId update_distance = v->distance + 1;
        update = new Update_SSSP(e->target, update_distance);
    }
    return update;
}

void apply_one_update(Update_SSSP *update, Vertex_SSSP *dst_vertex) {
    if (update->is_valid()) {
        if (update->distance < dst_vertex->distance) {
            dst_vertex->distance = update->distance;
        }
    }
}

int main(int argc, char **argv) {
    std::string input_file_name = std::string(argv[1]);
    int number_of_partitions = atoi(argv[2]);
    int input_format = atoi(argv[3]);
    int number_of_threads = atoi(argv[4]);
    int number_of_iterations = atoi(argv[5]);
    int start_vertex_id = atoi(argv[6]);
    ::start_vertex_id = start_vertex_id;
    Engine e(input_file_name, number_of_partitions, input_format, number_of_threads);

    // get running time (wall time)
    auto start = std::chrono::high_resolution_clock::now();

    std::cout << "--------------------Init Vertex--------------------" << std::endl;
    e.init_vertex<Vertex_SSSP>(init);

    int num_iters = number_of_iterations;
    for (int i = 0; i < num_iters; i++) {
        std::cout << "--------------------Iteration " << i << "--------------------" << std::endl;
        Scatter<Vertex_SSSP, Update_SSSP> scatter_phase(e);
        Update_Stream in_stream = scatter_phase.scatter_with_vertex(generate_one_update);
        Gather<Vertex_SSSP, Update_SSSP> gather_phase(e);
        gather_phase.gather(in_stream, apply_one_update);

        Global_Info::delete_upstream(in_stream, e);
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    std::cout << "Finish sssp. Running time : " << diff.count() << " s\n";
}
