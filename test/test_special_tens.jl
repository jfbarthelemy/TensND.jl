@testsection "Special tensors" begin
    # Rotations
    θ, ϕ, ψ = symbols("θ ϕ ψ", real = true) ;
    cθ, cϕ, cψ, sθ, sϕ, sψ = symbols("cθ cϕ cψ sθ sϕ sψ", real = true) ;
    d = Dict(cos(θ) => cθ, cos(ϕ) => cϕ, cos(ψ) => cψ, sin(θ) => sθ, sin(ϕ) => sϕ, sin(ψ) => sψ) ;
    R = Tens(subs(rot3(θ, ϕ, ψ),d...)) ;
    R6 = invKM(subs(KM(rot6(θ, ϕ, ψ)),d...)) ;
    @test R6 == R ⊠ˢ R

end
