/*
 * engine.hpp
 *
 *  Created on: Mar 3, 2017
 *      Author: kai
 */

#ifndef CORE_ENGINE_HPP_
#define CORE_ENGINE_HPP_


#include "concurrent_queue.hpp"
#include "../struct/type.hpp"
#include "../utility/FileUtil.hpp"

//#include "../preprocessor/preproc.hpp"
//#include "../preprocessor/preprocessing.hpp"
#include "../preprocessor/preprocessing_new.hpp"


namespace RStream {

	class Engine {
	public:
		int num_threads;
		int num_write_threads;
		int num_exec_threads;

		std::string filename;
		int num_partitions;
//		std::vector<int> num_vertices;

		EdgeType edge_type;
		// sizeof each edge
		int edge_unit;

		int vertex_unit;

		int num_vertices;
		int num_vertices_per_part;

//		int* vertex_intervals;
		std::vector<std::pair<VertexId, VertexId>> vertex_intervals;

		static unsigned update_count;
		static unsigned aggregation_count;

		//for debugging
		static unsigned tuple_total;
		static unsigned tuple_auto;
		static unsigned tuple_long;
		static unsigned tuple_filter;

		Engine(std::string _filename, int num_parts, int input_format, int number_of_threads = 64);

		~Engine();

		//clean files added by Zhiqiang
		void clean_files();


		/* init vertex data*/
		template <typename VertexDataType>
		void init_vertex(std::function<void(char*, VertexId)> init) {
			vertex_unit = sizeof(VertexDataType);

			// a pair of <vertex_file, num_vertices>
//			concurrent_queue<std::pair<int, int>> * task_queue = new concurrent_queue<std::pair<int, int>>(num_partitions);

			// <vertex_file, num_vertices, start_vertex_id>
			concurrent_queue<std::tuple<int, VertexId, VertexId>> * task_queue = new concurrent_queue<std::tuple<int, VertexId, VertexId>>(num_partitions);

			for(int partition_id = 0; partition_id < num_partitions; partition_id++) {
				int perms = O_WRONLY;
				std::string vertex_file = filename + "." + std::to_string(partition_id) + ".vertex";
				int fd = open(vertex_file.c_str(), perms, S_IRWXU);
				if(fd < 0) {
					fd = creat(vertex_file.c_str(), S_IRWXU);
				}

				VertexId n_vertices = vertex_intervals[partition_id].second - vertex_intervals[partition_id].first + 1;
//				task_queue->push(std::make_pair(fd, n_vertices));
				task_queue->push(std::make_tuple(fd, n_vertices, vertex_intervals[partition_id].first));

			}

			// threads will load vertex and update, and apply update one by one
			std::vector<std::thread> threads;
			for(int i = 0; i < num_threads; i++)
				threads.push_back(std::thread(&Engine::init_produer<VertexDataType>, this, init, task_queue));

			// join all threads
			for(auto & t : threads)
				t.join();
		}

		/*compute out degree for each vertex*/
		template <typename VertexDataType>
		void compute_degree() {
			concurrent_queue<int> * task_queue = new concurrent_queue<int>(num_partitions);

			// push task into concurrent queue
			for(int partition_id = 0; partition_id < num_partitions; partition_id++) {
				task_queue->push(partition_id);
			}

			std::vector<std::thread> threads;
			for(int i = 0; i < num_threads; i++)
				threads.push_back(std::thread(&Engine::compute_degree_producer<VertexDataType>, this, task_queue));

			// join all threads
			for(auto & t : threads)
				t.join();

			delete task_queue;
		}



	private:

//		template <typename VertexDataType>
//		void init_produer(std::function<void(char*)> init, concurrent_queue<std::pair<int, int>> * task_queue);

		template <typename VertexDataType>
		void init_produer(std::function<void(char*, VertexId)> init, concurrent_queue<std::tuple<int, VertexId, VertexId>> * task_queue) {

			int fd = -1;
			VertexId num_vertex = 0, start_vertex = -1;
			auto one_task = std::make_tuple(fd, num_vertex, start_vertex);

//			while(task_queue->test_pop_atomic(pair)) {
			while(task_queue->test_pop_atomic(one_task)) {
//				int fd = pair.first;
//				int num_vertex = pair.second;
//				assert(fd > 0 && num_vertex > 0 );

				fd = std::get<0>(one_task);
				num_vertex = std::get<1>(one_task);
				start_vertex = std::get<2>(one_task);
				assert(fd > 0 && num_vertex > 0 && start_vertex >= 0);

				// size_t ok??
				size_t vertex_file_size = num_vertex * sizeof(VertexDataType);
				char * vertex_local_buf = new char[vertex_file_size];

				// for each vertex
//				for(size_t pos = 0; pos < vertex_file_size; pos += sizeof(VertexDataType)) {
//					init(vertex_local_buf + pos);
//				}

				VertexId counter = 0;
				// for each vertex
				for(size_t pos = 0; pos < vertex_file_size; pos += sizeof(VertexDataType)) {
					init(vertex_local_buf + pos, start_vertex + counter);
					counter++;
				}

				io_manager::write_to_file(fd, vertex_local_buf, vertex_file_size);

				delete[] vertex_local_buf;
				close(fd);
			}
		}

		template <typename VertexDataType>
		void load_vertices_hashMap(char* vertex_local_buf, const int vertex_file_size, std::unordered_map<VertexId, VertexDataType*> & vertex_map) {
			for(size_t off = 0; off < vertex_file_size; off += vertex_unit){
				VertexDataType* v = reinterpret_cast<VertexDataType*>(vertex_local_buf + off);
				vertex_map[v->id] = v;
			}
		}

		template <typename VertexDataType>
		void compute_degree_producer(concurrent_queue<int> * task_queue) {

			int partition_id = -1;
			while(task_queue->test_pop_atomic(partition_id)) {
				int fd_vertex = open((filename + "." + std::to_string(partition_id) + ".vertex").c_str(), O_RDWR);
				int fd_edge = open((filename + "." + std::to_string(partition_id)).c_str(), O_RDONLY);
				assert(fd_vertex > 0 && fd_edge > 0 );

				// get file size
				long vertex_file_size = io_manager::get_filesize(fd_vertex);
				long edge_file_size = io_manager::get_filesize(fd_edge);

				// vertex data fully loaded into memory
				char * vertex_local_buf = new char[vertex_file_size];
				io_manager::read_from_file(fd_vertex, vertex_local_buf, vertex_file_size, 0);
				std::unordered_map<VertexId, VertexDataType*> vertex_map;
				load_vertices_hashMap(vertex_local_buf, vertex_file_size, vertex_map);

				// streaming edges
//				char * edge_local_buf = (char *)memalign(PAGE_SIZE, IO_SIZE);
//				int streaming_counter = edge_file_size / IO_SIZE + 1;
				char * edge_local_buf = (char *)memalign(PAGE_SIZE, IO_SIZE * sizeof(Edge));
				int streaming_counter = edge_file_size / (IO_SIZE * sizeof(Edge)) + 1;

				long valid_io_size = 0;
				long offset = 0;

				assert(edge_unit == sizeof(Edge));
				// for all streaming
				for(int counter = 0; counter < streaming_counter; counter++) {

					// last streaming
					if(counter == streaming_counter - 1)
						// TODO: potential overflow?
//						valid_io_size = edge_file_size - IO_SIZE * (streaming_counter - 1);
						valid_io_size = edge_file_size - IO_SIZE * sizeof(Edge) * (streaming_counter - 1);
					else
//						valid_io_size = IO_SIZE;
						valid_io_size = IO_SIZE * sizeof(Edge);

					assert(valid_io_size % sizeof(edge_unit) == 0);

					io_manager::read_from_file(fd_edge, edge_local_buf, valid_io_size, offset);
					offset += valid_io_size;

					for(long pos = 0; pos < valid_io_size; pos += edge_unit) {
						// get an edge
						Edge * e = (Edge*)(edge_local_buf + pos);
						assert(vertex_map.find(e->src) != vertex_map.end());
						VertexDataType * src_vertex = vertex_map.find(e->src)->second;
						src_vertex->degree++;
					}

				}

				//for debugging
//				for(size_t off = 0; off < vertex_file_size; off += vertex_unit){
//					VertexDataType* v = reinterpret_cast<VertexDataType*>(vertex_local_buf + off);
//					std::cout << *v << std::endl;
//				}

				// write updated vertex value to disk
				io_manager::write_to_file(fd_vertex, vertex_local_buf, vertex_file_size);

				// delete
				delete[] vertex_local_buf;
//				delete[] edge_local_buf;
				free(edge_local_buf);
				close(fd_vertex);
				close(fd_edge);
			}
		}

		inline bool file_exists(const std::string  filename) {
			struct stat buffer;
			return (stat(filename.c_str(), &buffer) == 0);
		}

		void read_meta_file(const std::string & filename);

		// Removes \n from the end of line
		inline void FIXLINE(char * s) {
			int len = (int) strlen(s)-1;
			if(s[len] == '\n') s[len] = 0;
		}
	};


}



#endif /* CORE_ENGINE_HPP_ */

