cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(AlphaZero)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall -pedantic")

if (${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_LINUX")
endif()

set(CMAKE_PREFIX_PATH /usr/local/libtorch)
set(Gomoku_PATH /usr/local/lib/libgomoku.a)

find_package(Torch REQUIRED)
find_package(cppzmq REQUIRED)

add_executable(main main.cpp)
add_executable(generate generate.cpp)
add_executable(train train.cpp)
add_executable(gpu2cpu gpu2cpu.cpp)

target_include_directories(main PRIVATE /usr/local/include ${PROJECT_SOURCE_DIR}/../external)
target_link_libraries(main "${TORCH_LIBRARIES}" /usr/local/lib/libzmq.a ${Gomoku_PATH})
set_property(TARGET main PROPERTY CXX_STANDARD 17)

target_include_directories(generate PRIVATE /usr/local/include ${PROJECT_SOURCE_DIR}/../external)
target_link_libraries(generate "${TORCH_LIBRARIES}" /usr/local/lib/libzmq.a ${Gomoku_PATH})
set_property(TARGET generate PROPERTY CXX_STANDARD 17)

target_include_directories(train PRIVATE /usr/local/include ${PROJECT_SOURCE_DIR}/../external)
target_link_libraries(train "${TORCH_LIBRARIES}" /usr/local/lib/libzmq.a ${Gomoku_PATH})
set_property(TARGET train PROPERTY CXX_STANDARD 17)

target_include_directories(gpu2cpu PRIVATE /usr/local/include ${PROJECT_SOURCE_DIR}/../external)
target_link_libraries(gpu2cpu "${TORCH_LIBRARIES}" ${Gomoku_PATH})
set_property(TARGET gpu2cpu PROPERTY CXX_STANDARD 17)