// ---- benchmark metadata ----
// benchmark=mock_baddefault
// level=1
// component=Test/Mock
// description=Mock benchmark for wrapper tests with a documented default that --check rejects
// tags=test;mock
// version=0.1.0
// precisions=fp64,fp32,fp16,int32
// extra_build_flags=
//
// # param.<name>=<default>|<valid values>|<effect>
// param.duration=required|> 0 (seconds)|Length of the timed loop
// param.ops_per_element=0|>= 1|Work per element
// param.precision=fp64|see precisions|Data type
// ---- end metadata ----

// Mock for tests: the "binary" is tests/mock/bin/mock_baddefault, a shell script that
// calls tests/mock/mockbench.sh with behavior 'ok'.
