find_program(SAF_GLSLC_EXECUTABLE glslc)
find_program(SAF_SLANGC_EXECUTABLE slangc)

# Compiles each source to <target's runtime output directory>/shaders/<file name>.spv.
function(saf_add_shaders target)
    cmake_parse_arguments(PARSE_ARGV 1 arg "" "" "SOURCES")
    if(arg_UNPARSED_ARGUMENTS OR NOT arg_SOURCES)
        message(FATAL_ERROR "saf_add_shaders: expected saf_add_shaders(${target} SOURCES <files...>)")
    endif()

    set(glslExtensions .vert .tesc .tese .geom .frag .comp .mesh .task .rgen .rint .rahit .rchit .rmiss .rcall)

    get_target_property(runtimeDir ${target} RUNTIME_OUTPUT_DIRECTORY)
    if(NOT runtimeDir)
        message(FATAL_ERROR "saf_add_shaders: set RUNTIME_OUTPUT_DIRECTORY on ${target} first")
    endif()

    # Multi-config generators put the target in a per-config subdirectory unless the directory has a generator expression
    get_property(multiConfig GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
    set(configSuffix)
    if(multiConfig)
        set(configSuffix .$<CONFIG>)
        if(NOT runtimeDir MATCHES "\\$<")
            string(APPEND runtimeDir /$<CONFIG>)
        endif()
    endif()

    set(depfileDir ${CMAKE_CURRENT_BINARY_DIR}/shaders)
    file(MAKE_DIRECTORY ${depfileDir})

    set(outputs)
    set(names)
    foreach(source IN LISTS arg_SOURCES)
        cmake_path(ABSOLUTE_PATH source OUTPUT_VARIABLE sourcePath)
        cmake_path(GET source FILENAME name)
        cmake_path(GET source EXTENSION LAST_ONLY extension)
        if(name IN_LIST names)
            message(FATAL_ERROR "saf_add_shaders: two sources are named ${name}; both would compile to shaders/${name}.spv")
        endif()
        list(APPEND names ${name})
        set(output ${runtimeDir}/shaders/${name}.spv)
        set(depfile ${depfileDir}/${name}${configSuffix}.d)

        if(extension IN_LIST glslExtensions)
            if(NOT SAF_GLSLC_EXECUTABLE)
                message(FATAL_ERROR "saf_add_shaders: glslc not found, needed for ${source}")
            endif()
            set(command ${SAF_GLSLC_EXECUTABLE} --target-env=vulkan1.3 -MD -MF ${depfile} -o ${output} ${sourcePath})
        elseif(extension STREQUAL ".slang")
            if(NOT SAF_SLANGC_EXECUTABLE)
                message(FATAL_ERROR "saf_add_shaders: slangc not found, needed for ${source}")
            endif()
            # Adding -fp-mode precise instead of fast decorates every float operation NoContraction if deterministic and correct fp math is required..
            # That makes results identical wherever the same expression is inlined, at some speed, for all shaders.
            set(command ${SAF_SLANGC_EXECUTABLE} ${sourcePath} -target spirv -fvk-use-entrypoint-name  -O3 -fp-mode fast -depfile ${depfile} -o ${output})
        else()
            message(FATAL_ERROR "saf_add_shaders: no shader compiler for '${extension}' (${source})")
        endif()

        add_custom_command(
            OUTPUT ${output}
            COMMAND ${CMAKE_COMMAND} -E make_directory ${runtimeDir}/shaders
            COMMAND ${command}
            MAIN_DEPENDENCY ${sourcePath}
            DEPFILE ${depfile}
            COMMENT "Compiling shader ${name}"
            VERBATIM
        )
        list(APPEND outputs ${output})
    endforeach()

    add_custom_target(${target}-shaders DEPENDS ${outputs})
    add_dependencies(${target} ${target}-shaders)
    install(FILES ${outputs} DESTINATION bin/shaders)
endfunction()
