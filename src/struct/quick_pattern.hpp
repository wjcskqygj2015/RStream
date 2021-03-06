/*
 * quick_pattern.hpp
 *
 *  Created on: Aug 4, 2017
 *      Author: icuzzq
 */

#ifndef SRC_CORE_QUICK_PATTERN_HPP_
#define SRC_CORE_QUICK_PATTERN_HPP_

#include "type.hpp"

namespace RStream {

class Quick_Pattern {

	friend std::ostream & operator<<(std::ostream & strm, const Quick_Pattern& quick_pattern);

public:
	Quick_Pattern(unsigned int size_of_tuple);

//	Quick_Pattern(std::vector<Element_In_Tuple>& t){
//		tuple = t;
//	}

	~Quick_Pattern();

	//operator for map
	bool operator==(const Quick_Pattern& other) const;

	unsigned int get_hash() const;


	Element_In_Tuple& at(unsigned int index) const;

//	inline void push(Element_In_Tuple& element){
//		tuple.push_back(element);
//	}

//	inline std::vector<Element_In_Tuple> get_tuple() const {
//		return tuple;
//	}

	inline unsigned int get_size() const {
		return size;
	}

	inline Element_In_Tuple* get_elements(){
		return elements;
	}

	inline void clean(){
		delete[] elements;
	}

private:
//	std::vector<Element_In_Tuple> tuple;
	unsigned int size;
	Element_In_Tuple* elements;

};

}

namespace std {
	template<>
	struct hash<RStream::Quick_Pattern> {
		std::size_t operator()(const RStream::Quick_Pattern& qp) const {
			//simple hash
			return std::hash<int>()(qp.get_hash());
		}
	};
}


#endif /* SRC_CORE_QUICK_PATTERN_HPP_ */
