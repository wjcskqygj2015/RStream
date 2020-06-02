/*
 * type.hpp
 *
 *  Created on: Mar 6, 2017
 *      Author: kai
 */

#ifndef CORE_TYPE_HPP_
#define CORE_TYPE_HPP_

#include "../common/RStreamCommon.hpp"

#include "defs.hh"
#include "graph.hh"
#include "timer.hh"
#include "utils.hh"
#include "bignum.hh"
#include "uintseqhash.hh"

namespace RStream {


typedef unsigned Update_Stream;
typedef unsigned Aggregation_Stream;
typedef int VertexId;
typedef float Weight;
typedef unsigned char BYTE;

enum class FORMAT {
	EdgeList = 0,
	AdjList = 1
};

enum class EdgeType {
	NO_WEIGHT = 0,
	WITH_WEIGHT = 1,
	Labeled = 2
};

inline std::ostream& operator<<(std::ostream& o, const EdgeType& c)
{
    if(c == EdgeType::NO_WEIGHT){
    	o << "NO_WEIGHT";
    }
    else if(c == EdgeType::WITH_WEIGHT){
    	o << "WITH_WEIGHT";
    }
    else if(c == EdgeType::Labeled) {
    	o << "LabeledEdge";
    }
    else{
    	std::cout << "wrong edge type!!!" << std::endl;
    	throw std::exception();
    }
    return o;
}

struct Edge {
	VertexId src;
	VertexId target;

	Edge(VertexId _src, VertexId _target) : src(_src), target(_target) {}
	Edge() : src(0), target(0) {}
	~Edge(){}

	std::string toString(){
		return "(" + std::to_string(src) + ", " + std::to_string(target) + ")";
	}

	void swap(){
		if (src > target) {
			VertexId tmp = src;
			src = target;
			target = tmp;
		}
	}

};

class EdgeComparator{
public:
	int operator()(const Edge& oneEdge, const Edge& otherEdge){
		if(oneEdge.src == otherEdge.src){
			return oneEdge.target > otherEdge.target;
		}
		else{
			return oneEdge.src > otherEdge.src;
		}
	}
};

struct LabeledEdge {
	VertexId src;
	VertexId target;
	BYTE src_label;
	BYTE target_label;

	LabeledEdge(VertexId _src, VertexId _target, BYTE _src_label, BYTE _target_label) : src(_src), target(_target), src_label(_src_label), target_label(_target_label) {}
	LabeledEdge() : src(0), target(0), src_label(0), target_label(0) {}
}__attribute__((__packed__));

struct WeightedEdge {
	VertexId src;
	VertexId target;
	Weight weight;

	WeightedEdge(VertexId _src, VertexId _target, Weight _weight) : src(_src), target(_target), weight(_weight) {}
	WeightedEdge() : src(0), target(0), weight(0.0f) {}

	std::string toString(){
		return "(" + std::to_string(src) + ", " + std::to_string(target) + std::to_string(weight) + ")";
	}
}__attribute__((__packed__));

//struct Vertex_Interval {
//	VertexId start;
//	VertexId end;
//};

inline std::ostream & operator<<(std::ostream & strm, const WeightedEdge& edge){
	strm << "(" << edge.src << ", " << edge.target << ", " << edge.weight << ")";
	return strm;
}

inline std::ostream & operator<<(std::ostream & strm, const Edge& edge){
	strm << "(" << edge.src << ", " << edge.target  << ")";
	return strm;
}

inline std::ostream & operator<<(std::ostream & strm, const LabeledEdge& edge){
	strm << "(" << edge.src << ", " << (int)edge.src_label << " - " << edge.target << ", " << (int)edge.target_label << ")";
	return strm;
}


struct BaseUpdate {
	VertexId target;

	BaseUpdate(){};
	BaseUpdate(VertexId _target) : target(_target) {};

	std::string toString(){
		return std::to_string(target);
	}

};

struct BaseVertex {
	VertexId id;
};

//struct Update_Stream {
//	unsigned update_filename;
//};

inline std::ostream & operator<<(std::ostream & strm, const BaseUpdate& up){
	strm << "(" << up.target << ")";
	return strm;
}


/*
 *  Graph mining support. Join on all keys for each vertex tuple.
 *  Each element in the tuple contains 8 bytes, first 4 bytes is vertex id,
 *  second 4 bytes contains edge label(1byte) + vertex label(1byte) + history info(1byte).
 *  History info is used to record subgraph structure.
 *
 *
 *  [ ] [ ] [ ] [ ] || [ ] [ ] [ ] [ ]
 *    vertex id        idx  el  vl info
 *     4 bytes          1   1   1    1
 *
 * */
struct Element_In_Tuple {
	VertexId vertex_id;
	BYTE key_index;
	BYTE edge_label;
	BYTE vertex_label;
	BYTE history_info;

	Element_In_Tuple(){

	}

	Element_In_Tuple(VertexId _vertex_id, BYTE _edge_label, BYTE _vertex_label) :
		vertex_id(_vertex_id), key_index(0), edge_label(_edge_label), vertex_label(_vertex_label), history_info(0) {

	}

	Element_In_Tuple(VertexId _vertex_id, BYTE _edge_label, BYTE _vertex_label, BYTE _history) :
				vertex_id(_vertex_id), key_index(0), edge_label(_edge_label), vertex_label(_vertex_label), history_info(_history) {

	}

	Element_In_Tuple(VertexId _vertex_id, BYTE _key_index, BYTE _edge_label, BYTE _vertex_label, BYTE _history) :
		vertex_id(_vertex_id), key_index(_key_index), edge_label(_edge_label), vertex_label(_vertex_label), history_info(_history) {

	}

	~Element_In_Tuple(){

	}

	inline void set_vertex_id(VertexId new_id){
		vertex_id = new_id;
	}

	inline int cmp(const Element_In_Tuple& other) const {
		//compare vertex id
		if(vertex_id < other.vertex_id){
			return -1;
		}
		if(vertex_id > other.vertex_id){
			return 1;
		}

		//compare history info
		if(history_info < other.history_info){
			return -1;
		}
		if(history_info > other.history_info){
			return 1;
		}

		//compare vertex label
		if(vertex_label < other.vertex_label){
			return -1;
		}
		if(vertex_label > other.vertex_label){
			return 1;
		}

		//compare edge label
		if(edge_label < other.edge_label){
			return -1;
		}
		if(edge_label > other.edge_label){
			return 1;
		}

		//compare index
		if(key_index < other.key_index){
			return -1;
		}
		if(key_index > other.key_index){
			return 1;
		}

		return 0;
	}
};

inline std::ostream & operator<<(std::ostream & strm, const Element_In_Tuple& element){
	strm << "[" << element.vertex_id << ", " << (int)element.key_index << ", " << (int)element.edge_label << ", "
			<< (int)element.vertex_label << ", " << (int)element.history_info << "]";
	return strm;
}

inline std::ostream & operator<<(std::ostream & strm, const std::vector<Element_In_Tuple>& tuple){
	if(tuple.empty()){
		strm << "(empty)";
		return strm;
	}

	strm << "(";
	for(auto it = tuple.begin(); it != tuple.end() - 1; ++it){
		strm << (*it) << ", ";
	}
	strm << tuple.back();
	strm << ")";
	return strm;
}


struct Base_Element {
	VertexId id;

//	BaseElement(){}

	Base_Element(VertexId _vertex_id) : id(_vertex_id){}
	~Base_Element(){}

};

inline std::ostream & operator<<(std::ostream & strm, const Base_Element& element){
	strm << "[" << element.id << "]";
	return strm;
}



}





#endif /* CORE_TYPE_HPP_ */
