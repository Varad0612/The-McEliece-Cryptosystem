add_lab("RREF_test")
add_lab_solution("RREF_test" ${CMAKE_CURRENT_LIST_DIR}/main.cu)
add_generator("RREF_test" ${CMAKE_CURRENT_LIST_DIR}/dataset_generator.cpp)
