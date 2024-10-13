project('dab-reconstruct', 'c')
liquid = meson.get_compiler('c').find_library('liquid', has_headers: ['liquid/liquid.h'])

cc = meson.get_compiler('c')
m_dep = cc.find_library('m', required: false)
deps = [liquid, m_dep]

incdir = include_directories('include')
sources = files('src/nulldetector.c', 'src/syncsymbol.c')
dab_lib = static_library('dab', sources, include_directories: incdir)
executable('dab-reconstruct', 'main.c', dependencies: deps, include_directories: incdir, link_with: dab_lib)

test_workdir = meson.source_root()/'test'

test_nulldetector_exe = executable('test_nulldetector', 'test/test_nulldetector.c', dependencies: deps, include_directories: incdir, link_with: dab_lib)
test('test nulldetector', test_nulldetector_exe, workdir: test_workdir)

test_syncsymbol_exe = executable('test_syncsymbol', 'test/test_syncsymbol.c', dependencies: deps, include_directories: incdir, link_with: dab_lib)
test('test sync symbol', test_syncsymbol_exe, workdir: test_workdir)
