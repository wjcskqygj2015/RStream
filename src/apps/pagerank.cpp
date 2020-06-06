/*
 * pagerank.cpp
 *
 *  Created on: Mar 7, 2017
 *      Author: kai
 */

#include "../core/engine.hpp"
#include "../core/scatter.hpp"
#include "../core/gather.hpp"
#include "../core/global_info.hpp"
//#include "BaseApplication.hpp"

using namespace RStream;

struct Update_PR : BaseUpdate {
    float rank;

    Update_PR(int _target, float _rank) : BaseUpdate(_target), rank(_rank) {};

    Update_PR() : BaseUpdate(0), rank(0.0) {}

    std::string toString() {
        return "(" + std::to_string(target) + ", " + std::to_string(rank) + ")";
    }
}__attribute__((__packed__));

struct Vertex_PR : BaseVertex {
    int degree;
    float rank;
    float sum;
}__attribute__((__packed__));

//inline std::ostream & operator<<(std::ostream & strm, const Vertex& vertex){
//	strm << "[" << vertex.id << "," << vertex.degree << "]";
//	return strm;
//}

static double NUMBER_OF_VERTEXES = 0;

void init(char *data, VertexId id) {
    struct Vertex_PR *v = (struct Vertex_PR *) data;
    v->degree = 0;
    v->sum = 0;
    v->rank = 0.15f / NUMBER_OF_VERTEXES;
    v->id = id;
}

Update_PR *generate_one_update_init(Edge *e, Vertex_PR *v) {
    Update_PR *update = new Update_PR(e->target, (0.15f / NUMBER_OF_VERTEXES) / v->degree);
    return update;
}

Update_PR *generate_one_update(Edge *e, Vertex_PR *v) {
    Update_PR *update = new Update_PR(e->target, v->rank / v->degree);
    return update;
}

void apply_one_update(Update_PR *update, Vertex_PR *dst_vertex) {
    dst_vertex->sum += update->rank;
    dst_vertex->rank = (0.15 / NUMBER_OF_VERTEXES) + 0.85 * dst_vertex->sum;
}

int main(int argc, char **argv) {
//		Engine e("/home/icuzzq/Workspace/git/RStream/input/input_new.txt", 3, 6);
    std::string input_file_name = std::string(argv[1]);
    int number_of_partitions = atoi(argv[2]);
    int input_format = atoi(argv[3]);
    int number_of_threads = atoi(argv[4]);
    int number_of_iterations = atoi(argv[5]);
    int number_of_vertexes = atoi(argv[6]);
    NUMBER_OF_VERTEXES = number_of_vertexes;
    Engine e(input_file_name, number_of_partitions, input_format, number_of_threads);

    // get running time (wall time)
    auto start = std::chrono::high_resolution_clock::now();

    std::cout << "--------------------Init Vertex--------------------" << std::endl;
    e.init_vertex<Vertex_PR>(init);
    std::cout << "--------------------Compute Degree--------------------" << std::endl;
    e.compute_degree<Vertex_PR>();

    int num_iters = number_of_iterations;

    Scatter<Vertex_PR, Update_PR> scatter_phase(e);
    Gather<Vertex_PR, Update_PR> gather_phase(e);

    for (int i = 0; i < num_iters; i++) {
        std::cout << "--------------------Iteration " << i << "--------------------" << std::endl;

        Update_Stream in_stream;
        if (i == 0) {
            in_stream = scatter_phase.scatter_with_vertex(generate_one_update_init);
        } else {
            in_stream = scatter_phase.scatter_with_vertex(generate_one_update);
        }
//			Update_Stream in_stream = scatter_phase.scatter_with_vertex(generate_one_update);
//			Gather<Vertex_PR, Update_PR> gather_phase(e);

        gather_phase.gather(in_stream, apply_one_update);

        Global_Info::delete_upstream(in_stream, e);
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    std::cout << "Finish page rank. Running time : " << diff.count() << " s\n";

}



