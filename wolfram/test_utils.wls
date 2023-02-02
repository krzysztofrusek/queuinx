   (* Copyright 2023 Krzysztof Rusek

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. *)

(* http://xahlee.info/M/WolframLang_function_by_pattern.html *)

Clear["Global`*"]

PasProb[rho_, b_]:= Sum[rho^i,{i,0,b-1}]/Sum[rho^i,{i,0,b}] ;
PasProb[1, b_]:= b/(1+b) ;
PasProb2[lambda_,mu_, b_]:= PasProb[lambda/mu,b] ;
PasProbQ[q_]:= PasProb[Lambda[id]/mu,b]/.q ;
CumProd[x_]:= Most[FoldList[Times, 1, x]] ;

prec = 64
qs = {
    {b->10, mu->SetPrecision[20,prec], id->1},
    {b->5, mu->SetPrecision[20,prec], id->2},
    {b->10, mu->SetPrecision[30,prec], id->3},
    {b->5, mu->SetPrecision[20,prec], id->4},
    {b->5, mu->SetPrecision[20,prec], id->5}
}

flows = {10,1,5,20.0000000001}

(* qs = SetPrecision[qs,32]  *)
flows = SetPrecision[flows,prec] 

routes = {
    {1, 3, 2},
    {3, 2},
    {3, 5},
    {4, 3, 5}
}

sol = NSolve[Table[Lambda[qi]==Total[MapThread[Part,{
    CumProd /@ Map[PasProbQ/@qs[[#]]&, routes] flows,
    Map[ r|->FirstPosition[r,qi,{}],routes]
}],2],{qi,1,5}],
Table[Lambda[i],{i,1,5}]
]

Print[sol]

Export["sol.json","sol"->Table[ToString[i]->Lambda[i],{i,1,5}]/.sol]

pi[i_,rho_, b_]:= rho^i/Sum[rho^k,{k,0,b}]



w= Map[
    Mean[MixtureDistribution[Table[pi[n,Lambda[id]/mu,b]/.sol[[1]],{n,0,b-1}],Table[GammaDistribution[n, 1/mu],{n,1,b}]]] /.# &,
    qs] 

jitter = Map[
    Variance[MixtureDistribution[Table[pi[n,Lambda[id]/mu,b]/.sol[[1]],{n,0,b-1}],Table[GammaDistribution[n, 1/mu],{n,1,b}]]] /.# &,
    qs]

pass = Map[PasProbQ[#]/.sol[[1]]&,qs]
(* pass[[4]] = Sum[1^i,{i,0,b-1}]/Sum[1^i,{i,0,b}]/.qs[[4]] *)
flow=Map[Total[w[[#]]]&,routes]
flowjitter = Map[Total[jitter[[#]]]&,routes]
flowloss=Map[1-Fold[Times,pass[[#]]]&,routes]

(* Sum[k^2 pi[k,\[Rho],b],{k,0,b}]  //Export["tmp.tex", # ]& *)

Sum[k^2 pi[k,1,b],{k,0,b}]

flow//ToString
flowjitter//ToString
flowloss//ToString