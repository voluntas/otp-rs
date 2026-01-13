#!/usr/bin/env escript
%% -*- erlang -*-
%% Compare a single function between two BEAM files using beam_disasm.

-mode(compile).

main([BeamA, BeamB, FuncName, ArityStr]) ->
    Name = list_to_atom(FuncName),
    case string:to_integer(ArityStr) of
        {Arity, ""} ->
            compare(BeamA, BeamB, Name, Arity);
        _ ->
            io:format("invalid arity: ~s~n", [ArityStr]),
            halt(2)
    end;
main(_) ->
    io:format("usage: beam_fun_compare.escript <beam_a> <beam_b> <function> <arity>~n"),
    halt(2).

compare(BeamA, BeamB, Name, Arity) ->
    FunA = get_fun(BeamA, Name, Arity),
    FunB = get_fun(BeamB, Name, Arity),
    case strip_lines(FunA) =:= strip_lines(FunB) of
        true ->
            io:format("ok: ~p/~p~n", [Name, Arity]),
            halt(0);
        false ->
            io:format("diff: ~p/~p~n", [Name, Arity]),
            io:format("A=~p~n", [strip_lines(FunA)]),
            io:format("B=~p~n", [strip_lines(FunB)]),
            halt(1)
    end.

get_fun(Beam, Name, Arity) ->
    case beam_disasm:file(Beam) of
        {beam_file, _Mod, _Exports, _Attrs, _CompileInfo, Funcs} ->
            case [Instrs || {function, N, A, _Entry, Instrs} <- Funcs,
                           N =:= Name, A =:= Arity] of
                [Instrs|_] -> Instrs;
                [] ->
                    io:format("function not found: ~p/~p in ~s~n", [Name, Arity, Beam]),
                    halt(3)
            end;
        Error ->
            io:format("beam_disasm failed for ~s: ~p~n", [Beam, Error]),
            halt(4)
    end.

strip_lines([{'line', _}|T]) -> strip_lines(T);
strip_lines([H|T]) -> [H|strip_lines(T)];
strip_lines([]) -> [].
