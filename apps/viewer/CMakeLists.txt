#
# Copyright Regents of the University of Minnesota, 2015.  This software is released under the following license: http://opensource.org/licenses/GPL-2.0.
# Source code originally developed at the University of Minnesota Interactive Visualization Lab (http://ivlab.cs.umn.edu).
#
# Code author(s):
# 		Dan Orban (dtorban)
#

cmake_minimum_required (VERSION 2.8.2)
set (CMAKE_VERBOSE_MAKEFILE TRUE)

project (viewer)

#------------------------------------------
# Define the source and header files
#------------------------------------------

set (SOURCEFILES
source/ViewerApp.cpp
source/ViewerAppScene.cpp
source/main.cpp
)

set (HEADERFILES
include/ViewerApp.h
include/ViewerAppScene.h
)

source_group("Header Files" FILES ${HEADERFILES})

#------------------------------------------
# Include Directories
#------------------------------------------
include_directories (
  .
  ..
  include
  ../../core/include
  ../../gpu/include
  ../../vis/vrbase/include
  ../../vis/partflow/include
)

set(MinVR_DIR $ENV{MinVR_DIR})
find_package(MinVR COMPONENTS MVRCore AppKit_GLFW REQUIRED)
include_directories(${MinVR_INCLUDE_DIRS})

############################################################
#OS Specific flags should go here
############################################################

if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  set(CMAKE_XCODE_ATTRIBUTE_CLANG_CXX_LANGUAGE_STANDARD "c++11")
  set(CMAKE_XCODE_ATTRIBUTE_CLANG_CXX_LIBRARY "libc++")
    find_library(COCOA_LIB Cocoa)
    find_library(IOKIT_LIB IOKit)
    find_library(CORE_FOUNDATION_FRAMEWORK CoreFoundation)
    find_library(CORE_VIDEO_FRAMEWORK CoreVideo)
    set(LIBS_ALL ${LIBS_ALL} ${COCOA_LIB} ${IOKIT_LIB} ${CORE_FOUNDATION_FRAMEWORK} ${CORE_VIDEO_FRAMEWORK})
    message(STATUS "${CORE_VIDEO_FRAMEWORK}")

endif()

if (${CMAKE_SYSTEM_NAME} MATCHES "Linux")
  # Linux specific code
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11") 

	set(Lib_Config "none")
	set(MinVR_LIBS)
	FOREACH (LIB ${MinVR_LIBRARIES})
	  if("${LIB}" STREQUAL "debug" OR "${LIB}" STREQUAL "optimized")
	    set(Lib_Config ${LIB})
	  else()
	    list(APPEND MinVR_LIBS ${LIB})
	    if("${Lib_Config}" STREQUAL "debug" OR "${Lib_Config}" STREQUAL "optimized")
	      list(APPEND MinVR_LIBS ${Lib_Config})
	    endif()
	    set(Lib_Config none)
	  endif()
	ENDFOREACH(LIB)
	LIST(REVERSE MinVR_LIBS)

	set(MinVR_LIBRARIES)

	FOREACH (LIB ${MinVR_LIBS})
	  list(APPEND MinVR_LIBRARIES ${LIB})
	ENDFOREACH(LIB)	

	#find_package(Threads)
	#find_package(X11)
	#set(LIBS_ALL ${LIBS_ALL} ${CMAKE_THREAD_LIBS_INIT} rt Xrandr Xxf86vm Xi m ${X11_LIBRARIES})
endif (${CMAKE_SYSTEM_NAME} MATCHES "Linux")

if (${CMAKE_SYSTEM_NAME} MATCHES "Windows")
  # Windows specific code

endif (${CMAKE_SYSTEM_NAME} MATCHES "Windows")

if (MSVC)
	# Windows specific code
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP")
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /MP")

endif (MSVC)

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

add_executable(${PROJECT_NAME} ${HEADERFILES} ${SOURCEFILES})
target_link_libraries(${PROJECT_NAME} partflow partflow_vis vrbase ${MinVR_LIBRARIES} ${All_LIB})

#------------------------------------------
# Install Target
#------------------------------------------

install( TARGETS ${PROJECT_NAME}
         LIBRARY DESTINATION lib
         ARCHIVE DESTINATION lib
         RUNTIME DESTINATION bin)
