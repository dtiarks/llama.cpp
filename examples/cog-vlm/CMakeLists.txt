add_library(cog-vlm OBJECT
            cog-vlm.cpp
            cog-vlm.h
            eva-clip.cpp
            eva-clip.h
            )

target_link_libraries(cog-vlm PRIVATE ggml llama ${CMAKE_THREAD_LIBS_INIT})

target_include_directories(cog-vlm PUBLIC .)
target_include_directories(cog-vlm PUBLIC ../..)
target_include_directories(cog-vlm PUBLIC ../../common)

target_compile_features(cog-vlm PRIVATE cxx_std_11)

add_library(llava_static STATIC $<TARGET_OBJECTS:llava>)
if (BUILD_SHARED_LIBS)
    set_target_properties(llava PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_compile_definitions(llava PRIVATE LLAMA_SHARED LLAMA_BUILD)
    add_library(llava_shared SHARED $<TARGET_OBJECTS:llava>)
    target_link_libraries(llava_shared PRIVATE ggml llama ${CMAKE_THREAD_LIBS_INIT})
    install(TARGETS llava_shared LIBRARY)
endif()

if (NOT MSVC)
    target_compile_options(llava PRIVATE -Wno-cast-qual) # stb_image.h
endif()

if(TARGET BUILD_INFO)
    add_dependencies(cog-vlm BUILD_INFO)
endif()

set(TARGET cog-vlm-cli)
add_executable(cog-vlm-cli cog-vlm-cli.cpp)
install(TARGETS cog-vlm-cli RUNTIME)
target_link_libraries(cog-vlm-cli PRIVATE common cog-vlm ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(cog-vlm PRIVATE cxx_std_11)
