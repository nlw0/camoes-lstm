# Character-level LSTM on Os Lusíadas.
#
# The previous script fed each window as one matrix whose second dimension was
# time. Flux's LSTM reads that axis as the batch, so the net never learned
# across characters and samples came out as noise. This version feeds
# (vocab, time, batch), trains with truncated BPTT, checkpoints, and samples.
#
# On the pod, system CUDA on LD_LIBRARY_PATH clashes with CUDA.jl's libraries.
# Launch with the driver libs only:
#   export LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64
#   julia --project=/root/.julia/environments/v1.13 camoes.jl
#   julia --project=/root/.julia/environments/v1.13 camoes.jl smoke
#   julia --project=/root/.julia/environments/v1.13 camoes.jl sample
#   julia --project=/root/.julia/environments/v1.13 camoes.jl 40

ENV["MLDATADEVICES_SILENCE_WARN_NO_GPU"] = "1"
ENV["JULIA_CUDA_SILENT"] = "true"

using CUDA
using cuDNN
using Flux
using Downloads
using Random
using StatsBase
using Serialization
using Statistics
import Optimisers

const DATA_DIR = get(ENV, "CAMOES_DIR", "/workspace/camoes")
const CKPT_PATH = joinpath(DATA_DIR, "latest.jls")
const TEXT_URL = "https://www.gutenberg.org/files/3333/3333-0.txt"

struct CharLSTM
    rnn1
    rnn2
    dense
end

Flux.@layer CharLSTM

function CharLSTM(vocab::Int, hidden::Int)
    CharLSTM(
        LSTM(vocab => hidden; return_state=true),
        LSTM(hidden => hidden; return_state=true),
        Dense(hidden => vocab),
    )
end

# Full window from a zero state. Matches how each training batch is drawn.
function (m::CharLSTM)(x)
    h1, _ = m.rnn1(x)
    h2, _ = m.rnn2(h1)
    return m.dense(h2)
end

function nparams(m)
    sum(length, Optimisers.trainables(m))
end

scalar(x) = x isa Number ? Float64(x) : Float64(x[])

function load_lusiadas(dir)
    mkpath(dir)
    path = joinpath(dir, "lusiadas.txt")
    if !isfile(path)
        println("Downloading Os Lusíadas...")
        flush(stdout)
        Downloads.download(TEXT_URL, path)
    end
    raw = read(path, String)
    start_at = findfirst("As armas e os barões", raw)
    end_at = findfirst("*** END OF THE PROJECT GUTENBERG", raw)
    if isnothing(start_at)
        error("could not find the opening line in $(path)")
    end
    stop = isnothing(end_at) ? lastindex(raw) : prevind(raw, end_at[1])
    text = replace(raw[start_at[1]:stop], '\r' => "")
    return text
end

function encode_text(text)
    chars = collect(text)
    alphabet = sort(unique(chars))
    char_to_ix = Dict(c => i for (i, c) in enumerate(alphabet))
    tokens = Int32[char_to_ix[c] for c in chars]
    return tokens, alphabet
end

function make_batch(tokens::Vector{Int32}, seqlen::Int, batch::Int)
    maxstart = length(tokens) - seqlen - 1
    maxstart >= 1 || error("text is shorter than one window")
    starts = rand(1:maxstart, batch)
    x = Matrix{Int32}(undef, seqlen, batch)
    y = Matrix{Int32}(undef, seqlen, batch)
    @inbounds for b in 1:batch
        s = starts[b]
        for t in 1:seqlen
            x[t, b] = tokens[s + t - 1]
            y[t, b] = tokens[s + t]
        end
    end
    return x, y
end

function onehot_batch(idx::AbstractMatrix{<:Integer}, vocab::Int)
    time, batch = size(idx)
    x = zeros(Float32, vocab, time, batch)
    @inbounds for b in 1:batch, t in 1:time
        x[idx[t, b], t, b] = 1.0f0
    end
    return x
end

# One-hot is built outside the gradient. Zygote refuses the in-place fill.
function prepare_batch(tokens, vocab, seqlen, batch)
    x_idx, y_idx = make_batch(tokens, seqlen, batch)
    x = gpu(onehot_batch(x_idx, vocab))
    y = gpu(y_idx)
    return x, y
end

function train_step!(model, opt_state, tokens, vocab, seqlen, batch)
    x, y = prepare_batch(tokens, vocab, seqlen, batch)
    loss, grads = Flux.withgradient(model) do m
        Flux.logitcrossentropy(m(x), y)
    end
    Flux.update!(opt_state, model, grads[1])
    return scalar(loss)
end

function eval_loss(model, tokens, vocab, seqlen, batch; nbatch=4)
    total = 0.0
    for _ in 1:nbatch
        x, y = prepare_batch(tokens, vocab, seqlen, batch)
        total += scalar(Flux.logitcrossentropy(model(x), y))
    end
    return total / nbatch
end

function move_state(state)
    map(state) do x
        x isa AbstractArray ? gpu(x) : x
    end
end

function generate(model, alphabet, seed::String; n::Int=400, temperature::Float32=0.8f0)
    char_to_ix = Dict(c => i for (i, c) in enumerate(alphabet))
    for c in seed
        haskey(char_to_ix, c) || error("seed character $(repr(c)) is not in the vocabulary")
    end
    vocab = length(alphabet)
    s1 = move_state(Flux.initialstates(model.rnn1))
    s2 = move_state(Flux.initialstates(model.rnn2))
    function step(ix, s1, s2)
        x = zeros(Float32, vocab, 1)
        x[ix, 1] = 1.0f0
        h1, s1 = model.rnn1(gpu(x), s1)
        h2, s2 = model.rnn2(h1, s2)
        logits = model.dense(h2)
        return vec(cpu(logits)), s1, s2
    end
    # Prime on the seed. The last call is the distribution for the next char.
    logits = nothing
    for c in seed
        logits, s1, s2 = step(char_to_ix[c], s1, s2)
    end
    out = collect(seed)
    for _ in 1:n
        probs = softmax(logits ./ temperature)
        # softmax can undershoot 0 by a hair on Float32
        probs = max.(probs, 0)
        probs ./= sum(probs)
        next_ix = sample(1:vocab, Weights(probs))
        push!(out, alphabet[next_ix])
        logits, s1, s2 = step(next_ix, s1, s2)
    end
    return String(out)
end

function save_checkpoint(path, model, meta)
    mkpath(dirname(path))
    host = cpu(model)
    serialize(path, (; state=Flux.state(host), meta))
    println("saved $(path)")
    flush(stdout)
end

function load_checkpoint(path, vocab, hidden)
    blob = deserialize(path)
    model = CharLSTM(vocab, hidden)
    Flux.loadmodel!(model, blob.state)
    return model, blob.meta
end

function logline(msg)
    println(msg)
    flush(stdout)
end

function main()
    mkpath(DATA_DIR)
    smoke = length(ARGS) >= 1 && ARGS[1] == "smoke"
    sample_only = length(ARGS) >= 1 && ARGS[1] == "sample"

    text = load_lusiadas(DATA_DIR)
    tokens, alphabet = encode_text(text)
    vocab = length(alphabet)

    hidden = smoke ? 64 : 512
    seqlen = smoke ? 32 : 128
    batch = smoke ? 8 : 64
    lr = 1e-3
    epochs = if smoke
        1
    elseif length(ARGS) >= 1 && ARGS[1] != "sample"
        parse(Int, ARGS[1])
    else
        60
    end

    if sample_only
        isfile(CKPT_PATH) || error("no checkpoint at $(CKPT_PATH)")
        blob = deserialize(CKPT_PATH)
        hidden = blob.meta.hidden
        model, meta = load_checkpoint(CKPT_PATH, vocab, hidden)
        model = gpu(model)
        logline("checkpoint step=$(meta.step) train_loss=$(meta.train_loss)")
        logline(generate(model, alphabet, "As armas e os barões"; n=500, temperature=0.75f0))
        return
    end

    split_at = clamp(floor(Int, length(tokens) * 0.92), seqlen + 2, length(tokens) - seqlen - 2)
    train_tokens = tokens[1:split_at]
    val_tokens = tokens[split_at+1:end]
    steps_per_epoch = max(1, fld(length(train_tokens) - 1, seqlen * batch))
    if smoke
        steps_per_epoch = 2
        epochs = 1
    end
    total_steps = epochs * steps_per_epoch

    use_gpu = CUDA.functional()
    logline("gpu=$(use_gpu) $(use_gpu ? CUDA.name(CUDA.device()) : "")")
    logline("chars=$(length(tokens)) vocab=$(vocab) train=$(length(train_tokens)) val=$(length(val_tokens))")
    logline("hidden=$(hidden) layers=2 seqlen=$(seqlen) batch=$(batch) steps_per_epoch=$(steps_per_epoch) epochs=$(epochs) lr=$(lr)")

    model = gpu(CharLSTM(vocab, hidden))
    logline("parameters=$(nparams(model))")
    x_probe, y_probe = prepare_batch(train_tokens, vocab, seqlen, batch)
    logline("batch x=$(typeof(x_probe)) $(size(x_probe)) y=$(typeof(y_probe)) $(size(y_probe))")
    x_probe isa CuArray || error("batch is not a CuArray. Flux gpu() is still on CPU.")
    opt = Optimisers.OptimiserChain(Optimisers.ClipNorm(1.0), Optimisers.Adam(lr))
    opt_state = Flux.setup(opt, model)

    step = 0
    t0 = time()
    last_loss = NaN
    for epoch in 1:epochs
        epoch_loss = 0.0
        for _ in 1:steps_per_epoch
            step += 1
            if step == 1
                logline("step 1 compiling (this can take several minutes)...")
            end
            last_loss = train_step!(model, opt_state, train_tokens, vocab, seqlen, batch)
            epoch_loss += last_loss
            if step == 1 || step % 10 == 0
                elapsed = time() - t0
                logline("step $(step)/$(total_steps)  loss=$(round(last_loss, digits=3))  $(round(step / elapsed, digits=2)) step/s")
            end
        end
        epoch_loss /= steps_per_epoch
        vloss = eval_loss(model, val_tokens, vocab, seqlen, batch)
        logline("epoch $(epoch)  train=$(round(epoch_loss, digits=3))  val=$(round(vloss, digits=3))  ppl=$(round(exp(epoch_loss), digits=2))")
        sample = generate(model, alphabet, "As armas e os barões"; n=smoke ? 80 : 320, temperature=0.75f0)
        logline("--- sample epoch $(epoch) ---")
        logline(sample)
        logline("--- end sample ---")
        meta = (;
            step, epoch, hidden, seqlen, batch, lr,
            train_loss=epoch_loss, val_loss=vloss,
            vocab, alphabet,
        )
        save_checkpoint(CKPT_PATH, model, meta)
        save_checkpoint(joinpath(DATA_DIR, "epoch-$(lpad(epoch, 3, '0')).jls"), model, meta)
    end
    logline("done steps=$(step) last_loss=$(round(last_loss, digits=3)) elapsed_min=$(round((time() - t0) / 60, digits=1))")
end

main()
