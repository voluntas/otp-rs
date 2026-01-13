-module(ext).
-export([f/0]).

f() -> erlang:abs(-1).
