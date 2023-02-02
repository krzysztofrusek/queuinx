   (* Copyright 2023 Krzysztof Rusek

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. 
   *)

a=25
mu=20
b=5
 \[Rho]=a/mu

pi0 = 1/Sum[ \[Rho]^i ,{i,0,b}]
p=Table[\[Rho]^i pi0,{i,0,b}]

logits = N[Table[Log[\[Rho]] i,{i,0,b}]]

pis = Exp[logits]
pis/Total[pis]

pi[i_,rho_, b_]:= rho^i/Sum[rho^k,{k,0,b}]


q=MixtureDistribution[Table[pi[n,a/mu,b],{n,0,b-1}],Table[GammaDistribution[n, 1/mu],{n,1,b}]]

N[Mean[q]]

N[Variance[q]]