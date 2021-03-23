cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_COLOR_MAKEFILE ON)
set(VERBOSE_BUILD ON)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(CMAKE_MACOSX_RPATH TRUE)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)


project(teachingkit-labs)

set(BUILD_LIBWB_LIBRARY ON CACHE BOOL "Turn on build of libwb library")
set(BUILD_SOLUTION ON CACHE BOOL "Turn on build of code solution")
# set(BUILD_DESCRIPTION OFF CACHE BOOL "Turn on build of lab description")
set(BUILD_DATASET_GENERATOR ON CACHE BOOL "Turn on building of dataset generator")
set(BUILD_TEMPLATE OFF CACHE BOOL "Turn on build of code templates")

if (BUILD_SOLUTION OR BUILD_TEMPLATE)
  find_package(CUDA REQUIRED)
else()
  find_package(CUDA)
endif ()

# allow C99 for cmake versions less than 3.1
macro(use_c99)
  if (CMAKE_VERSION VERSION_LESS "3.1")
    if (CMAKE_C_COMPILER_ID STREQUAL "GNU")
      set (CMAKE_C_FLAGS "--std=gnu99 ${CMAKE_C_FLAGS}")
    endif ()
  else ()
    set (CMAKE_C_STANDARD 99)
  endif ()
endmacro(use_c99)
use_c99()


set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --verbose")

# For some MPs it may be necessary to specify the compute capability of your
# NVIDIA GPU. In that case, simply uncomment the following two lines that start
# with 'set' and change the value of COMPUTE_CAPABILITY to one of the following
# hardware architectures: Tesla-class '11', '12', or '13'; Fermi-class '20'
# or '21'; Kepler-class '30', '32', '35'; or Maxwell-class '50' or '52'
set(COMPUTE_CAPABILITY 52)
# set(CUDA_SEPARABLE_COMPILATION TRUE)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --gpu-architecture;sm_${COMPUTE_CAPABILITY}")
# list(APPEND CUDA_NVCC_FLAGS -arch=sm_35)

# Set warning levels for host compilation
if (UNIX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-unused-function -Wno-unused-parameter -Wextra -std=c++11")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wno-unused-function -Wno-unused-parameter -Wextra")
    set(CUDA_PROPAGATE_HOST_FLAGS OFF)
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -Xcompiler -Wno-unused-function,-Wno-unused-parameter")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11")
endif ()

# Select the correct host compiler on OS X
if (APPLE)
    if (${CMAKE_SYSTEM_VERSION} LESS 12.0)
        # OS X 10.7 or less
        find_program(GCC_EXECUTABLE NAMES g++ PATHS ENV PATH)
        if (NOT GCC_EXECUTABLE)
            message(FATAL_ERROR "Could not find host compiler 'g++'")
        endif()
        set(CUDA_HOST_COMPILER ${GCC_EXECUTABLE})
    else ()
        # OS X 10.8+
        find_program(CLANG_EXECUTABLE NAMES clang++-3.6 clang++ PATHS ENV PATH)
        if (NOT CLANG_EXECUTABLE)
            message(FATAL_ERROR "Could not find host compiler 'clang++'")
        endif ()
        set(CUDA_HOST_COMPILER ${CLANG_EXECUTABLE})
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -isystem ${CUDA_INCLUDE_DIRS}")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -stdlib=libc++" )
    endif ()
endif ()

# Add debugging to CUDA NVCC flags (for NVidia's NSight tools)
set(CUDA_NVCC_FLAGS_DEBUG ${CUDA_NVCC_FLAGS_DEBUG} "-G –Xcompiler –rdynamic –lineinfo")

# allow unsupported compilers
set(CUDA_NVCC_FLAGS_DEBUG ${CUDA_NVCC_FLAGS_DEBUG} "-allow-unsupported-compiler")

# Set the correct linker options on UNIX platforms
# -lpthread inserted by ali akoglu  for compatiblity with elgato
if (UNIX)
    set(LINK_LIBRARIES "-lpthread -lm")
    if (APPLE)
        find_library(CUDA_LIBRARY NAMES CUDA)
        get_filename_component(FRAMEWORK_DIRECTORY ${CUDA_LIBRARY} DIRECTORY)
        set(LINK_LIBRARIES "${LINK_LIBRARIES} -F${FRAMEWORK_DIRECTORY} -framework CUDA")
    else ()
        include(CheckLibraryExists)
        check_library_exists(rt clock_gettime "time.h" HAVE_CLOCK_GETTIME)
        if (HAVE_CLOCK_GETTIME)
            set(LINK_LIBRARIES "${LINK_LIBRARIES} -lrt")
        else ()
            message(FATAL_ERROR "clock_gettime() not found")
        endif ()
    endif ()
endif ()

macro(add_lab_helper NAME SOURCES)
    cuda_add_executable(${NAME} ${SOURCES})
    target_link_libraries(${NAME} ${WBLIB} ${LINK_LIBRARIES})
endmacro(add_lab_helper)

macro(add_lab_solution NAME SOURCES)
    if (BUILD_SOLUTION)
        add_lab_helper("${NAME}_Solution" ${SOURCES})
    endif(BUILD_SOLUTION)
endmacro(add_lab_solution)


macro(add_lab_template NAME SOURCES)
    # template code is not meant to be compilable,
    # so we do not copile it. Enable the BUILD_TEMPLATE
    # flag during cmake generation to override this
    # behavior
    if (BUILD_TEMPLATE)
        add_lab_helper("${NAME}_Template" ${SOURCES})
    endif(BUILD_TEMPLATE)
endmacro(add_lab_template)

macro(add_generator NAME SOURCES)
  if (BUILD_DATASET_GENERATOR)
        add_executable("${NAME}_DatasetGenerator" ${SOURCES})
        set_property(TARGET "${NAME}_DatasetGenerator" PROPERTY CXX_STANDARD 11)
    target_link_libraries("${NAME}_DatasetGenerator" ${WBLIB} ${LINK_LIBRARIES})
  endif (BUILD_DATASET_GENERATOR)
endmacro(add_generator)

macro(add_lab NAME)
    if (BUILD_DESCRIPTION)
        find_program(PANDOC pandoc)
        set(MAKE_LAB_HANDOUT_PY "${PROJECT_SOURCE_DIR}/support/make_lab_handout.py")
        add_custom_target(
            "Build${NAME}Description"
            COMMAND python ${MAKE_LAB_HANDOUT_PY}
                            ${CMAKE_CURRENT_LIST_DIR}
                            ${PROJECT_SOURCE_DIR}/support
                            ${CMAKE_CURRENT_BINARY_DIR}
        )
        add_custom_target("${NAME}Description" ALL echo DEPENDS "Build${NAME}Description")
        set_property(DIRECTORY APPEND PROPERTY ADDITIONAL_MAKE_CLEAN_FILES ${NAME}.pdf)
        set_property(DIRECTORY APPEND PROPERTY ADDITIONAL_MAKE_CLEAN_FILES ${NAME}.docx)
    endif (BUILD_DESCRIPTION)
endmacro(add_lab)

include_directories(${PROJECT_SOURCE_DIR}/libwb)

if (BUILD_LIBWB_LIBRARY)
  include(libwb/sources.cmake)
  cuda_add_library(${WBLIB} ${LIBWB_SOURCE_FILES} STATIC)
  set_property(TARGET ${WBLIB} PROPERTY CXX_STANDARD 11)
endif (BUILD_LIBWB_LIBRARY)

include(hamc/sources.cmake)
