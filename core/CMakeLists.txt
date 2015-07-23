#
# Copyright Regents of the University of Minnesota, 2015.  This software is released under the following license: http://opensource.org/licenses/GPL-2.0.
# Source code originally developed at the University of Minnesota Interactive Visualization Lab (http://ivlab.cs.umn.edu).
#
# Code author(s):
# 		Dan Orban (dtorban)
#

cmake_minimum_required (VERSION 2.8.2)
set (CMAKE_VERBOSE_MAKEFILE TRUE)

project (partflow)

#------------------------------------------
# Define the source and header files
#------------------------------------------

set (SOURCEFILES
source/input/loaders/BlankLoader.cpp
source/input/loaders/BrickOfFloatLoader.cpp
source/input/loaders/CompositeDataLoader.cpp
source/input/loaders/ScaleLoader.cpp
source/input/loaders/VectorLoader.cpp
)

set (HEADERFILES
include/PFCore/input/loaders/BlankLoader.h
include/PFCore/input/loaders/BrickOfFloatLoader.h
include/PFCore/input/loaders/CompositeDataLoader.h
include/PFCore/input/loaders/ScaleLoader.h
include/PFCore/input/loaders/VectorLoader.h
include/PFCore/input/DataLoader.h
include/PFCore/math/v3.h
include/PFCore/math/v4.h
include/PFCore/math/vec_math.h
include/PFCore/partflow/ParticleSetView.h
include/PFCore/partflow/ParticleSet.h
include/PFCore/env_cuda.h
include/PFCore/env.h
)

source_group("Header Files" FILES ${HEADERFILES})

#------------------------------------------
# Include Directories
#------------------------------------------
include_directories (
  .
  ..
  include
)

#------------------------------------------
# Specific preprocessor defines
#------------------------------------------

# Windows Section #
if (MSVC)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
	# Tell MSVC to use main instead of WinMain for Windows subsystem executables
    set_target_properties(${WINDOWS_BINARIES} PROPERTIES LINK_FLAGS "/ENTRY:mainCRTStartup")
endif()

if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
	set(CMAKE_XCODE_ATTRIBUTE_CLANG_CXX_LANGUAGE_STANDARD "c++11")
	set(CMAKE_XCODE_ATTRIBUTE_CLANG_CXX_LIBRARY "libc++")
    set(All_LIB ${MinVR_LIBRARIES})
endif()

#------------------------------------------
# Set output directories to lib, and bin
#------------------------------------------
make_directory(${CMAKE_BINARY_DIR}/lib)
make_directory(${CMAKE_BINARY_DIR}/bin)
set (CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set (CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
foreach (CONF ${CMAKE_CONFIGURATION_TYPES})
	string (TOUPPER ${CONF} CONF)
	set (CMAKE_RUNTIME_OUTPUT_DIRECTORY_${CONF} ${CMAKE_BINARY_DIR}/bin)
	set (CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${CONF} ${CMAKE_BINARY_DIR}/lib)
	set (CMAKE_LIBRARY_OUTPUT_DIRECTORY_${CONF} ${CMAKE_BINARY_DIR}/lib)
endforeach(CONF CMAKE_CONFIGURATION_TYPES)

#------------------------------------------
# Build Target
#------------------------------------------

add_library ( ${PROJECT_NAME} ${HEADERFILES} ${SOURCEFILES} )
target_link_libraries(${PROJECT_NAME} ${All_LIB})

#------------------------------------------
# Install Target
#------------------------------------------

install(DIRECTORY include/ DESTINATION include)

install( TARGETS ${PROJECT_NAME}
         LIBRARY DESTINATION lib
         ARCHIVE DESTINATION lib
         RUNTIME DESTINATION bin)
