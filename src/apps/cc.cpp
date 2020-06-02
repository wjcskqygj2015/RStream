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

struct Update_CC : BaseUpdate {
    VertexId component;

    Update_CC(VertexId _target, VertexId _component) : BaseUpdate(_target), component(_component) {};

    Update_CC() : BaseUpdate(-1), component(-1) {}

    std::string toString() {
        return "(" + std::to_string(target) + ", " + std::to_string(component) + ")";
    }
}__attribute__((__packed__));

struct Vertex_CC : BaseVertex {
    VertexId component;
}__attribute__((__packed__));

void init(char *data, VertexId id) {
    struct Vertex_CC *v = (struct Vertex_CC *) data;
    v->id = id;
    v->component = id;
}

Update_CC *generate_one_update(Edge *e, Vertex_CC *v) {
    Update_CC *update = new Update_CC(e->target, v->component);
    return update;
}

void apply_one_update(Update_CC *update, Vertex_CC *dst_vertex) {
    if (update->component < dst_vertex->component) {
        dst_vertex->component = update->component;
    }
}

int main(int argc, char **argv) {
    std::string input_file_name = std::string(argv[1]);
    int number_of_partitions = atoi(argv[2]);
    int input_format = atoi(argv[3]);
    int number_of_threads = atoi(argv[4]);
    int number_of_iterations = atoi(argv[5]);
    Engine e(input_file_name, number_of_partitions, input_format, number_of_threads);

    // get running time (wall time)
    auto start = std::chrono::high_resolution_clock::now();

    std::cout << "--------------------Init Vertex--------------------" << std::endl;
    e.init_vertex<Vertex_CC>(init);

    int num_iters = number_of_iterations;
    for (int i = 0; i < num_iters; i++) {
        std::cout << "--------------------Iteration " << i << "--------------------" << std::endl;
        Scatter<Vertex_CC, Update_CC> scatter_phase(e);
        Update_Stream in_stream = scatter_phase.scatter_with_vertex(generate_one_update);
        Gather<Vertex_CC, Update_CC> gather_phase(e);
        gather_phase.gather(in_stream, apply_one_update);

        Global_Info::delete_upstream(in_stream, e);
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    std::cout << "Finish cc. Running time : " << diff.count() << " s\n";
}
