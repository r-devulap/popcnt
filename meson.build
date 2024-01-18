project('pocnt', 'cpp',
        version : '1.0.0',
        default_options : ['cpp_std=c++17'])
cpp = meson.get_compiler('cpp')

gbench_dep = dependency('benchmark', required : true, static: false)
benchexe = executable('benchexe',
                      'main.cpp',
                      dependencies : [gbench_dep],
                      link_args: ['-lbenchmark_main'],
                     )
