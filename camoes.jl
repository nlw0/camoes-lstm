using Flux
using Downloads
using Statistics
using Random
using CUDA

# 1. Download Os Lusíadas
url = "https://www.gutenberg.org/files/3333/3333-0.txt"
Downloads.download(url, "lusiadas.txt")

text = read("lusiadas.txt", String)
start_idx = findfirst("As armas e os barões", text)
end_idx   = findfirst("*** END OF THE PROJECT GUTENBERG", text)
text = isnothing(start_idx) ? text : text[start_idx[1]:(isnothing(end_idx) ? end : end_idx[1]-1)]

# Convert to Vector{Char} so indexing is safe
chars = collect(text)
println("Characters: ", length(chars))

# 2. Character vocabulary
alphabet = unique(chars)
char_to_ix = Dict(c => i for (i, c) in enumerate(alphabet))
ix_to_char = Dict(i => c for (i, c) in enumerate(alphabet))
vocab_size = length(alphabet)
println("Vocab size: ", vocab_size)

# 3. Model
hidden = 256
model = Chain(
    LSTM(vocab_size => hidden),
    LSTM(hidden => hidden),
    Dense(hidden => vocab_size)
)
model = CUDA.functional() ? model |> gpu : model
println("Using GPU: ", CUDA.functional())

# 4. Training
seqlen = 64
function get_batch(chars, batch_size=32)
    starts = rand(1:length(chars)-seqlen-1, batch_size)
    xs = [Flux.onehotbatch([char_to_ix[chars[i]] for i in s:s+seqlen-1], 1:vocab_size) for s in starts]
    ys = [Flux.onehotbatch([char_to_ix[chars[i]] for i in s+1:s+seqlen], 1:vocab_size) for s in starts]
    return xs, ys
end

opt_state = Flux.setup(Adam(0.001), model)

for epoch in 1:30
    xs, ys = get_batch(chars)
    l, grads = Flux.withgradient(model) do m
        sum(Flux.logitcrossentropy.(m.(xs), ys)) / length(xs)
    end
    Flux.update!(opt_state, model, grads[1])
    if epoch % 5 == 0
        println("Epoch $epoch  loss = $(round(l, digits=3))")
    end
end

# 5. Generate
function generate(model, seed::String, n::Int=250; temperature=0.8)
    Flux.reset!(model)
    result = seed
    for c in seed
        model(Flux.onehot(char_to_ix[c], 1:vocab_size) |> (CUDA.functional() ? gpu : identity))
    end
    for _ in 1:n
        input = Flux.onehot(char_to_ix[result[end]], 1:vocab_size) |> (CUDA.functional() ? gpu : identity)
        probs = softmax(model(input) ./ temperature)
        next_ix = sample(1:vocab_size, Weights(vec(probs)))
        result *= ix_to_char[next_ix]
    end
    return result
end

println("\n--- Sample ---")
println(generate(model, "As armas e os barões"))
