import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { visualizeTransformer } from '../api/api';
import { VisualizationResponse } from '../types';
import './Walkthrough.css';

interface Step {
  id: number;
  title: string;
  description: string;
  highlight: string[];
  dataFlow: { from: string; to: string }[];
  showData: (data: VisualizationResponse) => React.ReactNode;
}

const steps: Step[] = [
  {
    id: 0,
    title: 'Input Text',
    description: 'The journey begins here. We need to convert your text into something a neural network can actually understand. Since computers only speak numbers, we break down text into smaller units called tokens - think of them as the vocabulary building blocks.',
    highlight: ['inputs'],
    dataFlow: [],
    showData: (data) => (
      <div className="data-display">
        <div className="data-label">YOUR INPUT:</div>
        <div className="formula">"{data.state.tokens.join(' ')}"</div>
        <div className="data-label">Let's break this down</div>
        <div className="data-stat">Tokenization is how we split your text into pieces the model can work with. Imagine trying to teach someone a language - you don't hand them entire books, you start with individual words and common phrases.</div>
        <div className="data-label">HERE'S HOW YOUR TEXT SPLIT UP:</div>
        <div className="token-list">
          {data.state.tokens.map((token, i) => (
            <span key={i} className="token-badge">ID {i}: "{token}"</span>
          ))}
        </div>
        <div className="data-stat">That's {data.state.tokens.length} tokens total - each one gets a unique number</div>
        <div className="data-label">Why numbers though?</div>
        <div className="data-stat">Every token in the model's vocabulary gets assigned an ID number. So "the" might always be token 5, while "cat" could be 142. These aren't random - they come from a fixed dictionary the model learned during training, covering {data.metadata.vocabSize} possible tokens.</div>
        <div className="data-label">You might wonder: why not just use whole words?</div>
        <div className="data-stat">Great question. If we used complete words, we'd need millions of them in our vocabulary. Plus, what happens when someone types "ChatGPT" or makes a typo? The model would be stuck. Instead, we use subwords - smaller chunks that can combine to build any word imaginable.</div>
        <div className="data-label">The clever part</div>
        <div className="data-stat">Algorithms like Byte-Pair Encoding learn the most common subwords from massive amounts of text. So "learning" becomes ["learn", "##ing"] - the root word plus a common suffix. Even something ridiculous like "antidisestablishmentarianism" gets broken into manageable pieces.</div>
        <div className="data-label">What this gives us:</div>
        <div className="data-stat">A vocabulary of around 50,000 tokens that can represent essentially any text in any language. It's like having a finite set of Lego pieces that can build infinite structures. Pretty elegant, right?</div>
      </div>
    )
  },
  {
    id: 1,
    title: 'Input Embeddings',
    description: 'Now we convert those token IDs into something much richer: vectors that actually capture meaning. This is where the magic starts - similar words get similar vectors, letting the model understand relationships.',
    highlight: ['input-embeddings'],
    dataFlow: [{ from: 'inputs', to: 'input-embeddings' }],
    showData: (data) => {
      const firstToken = data.state.encoder[0]?.tokens[0];
      const secondToken = data.state.encoder[0]?.tokens[1];
      return (
        <div className="data-display">
          <div className="data-label">From IDs to vectors</div>
          <div className="data-stat">Each token ID gets transformed into a {data.metadata.embeddingDim}-dimensional vector - essentially a list of {data.metadata.embeddingDim} numbers. But here's what makes this special: these numbers actually encode meaning.</div>
          <div className="data-label">YOUR TOKEN: "{data.state.tokens[0]}"</div>
          <div className="vector-preview">
            [{firstToken?.embedding.slice(0, 8).map(v => v.toFixed(3)).join(', ')}, ...]
          </div>
          {secondToken && (
            <>
              <div className="data-label">YOUR TOKEN: "{data.state.tokens[1]}"</div>
              <div className="vector-preview">
                [{secondToken?.embedding.slice(0, 8).map(v => v.toFixed(3)).join(', ')}, ...]
              </div>
            </>
          )}
          <div className="data-label">Why does this matter?</div>
          <div className="data-stat">Once words are vectors, we can do math with meaning. "King" and "queen" end up with similar vectors because they share semantic properties - both relate to royalty. Same with "cat" and "dog" - they're much closer to each other than "cat" and "car", even though "cat" and "car" share more letters.</div>
          <div className="data-label">The mechanics are surprisingly simple</div>
          <div className="data-stat">We have a huge embedding table with {data.metadata.vocabSize} rows - one for each possible token. When we see token ID 5 (maybe "the"), we just look up row 5 and grab that {data.metadata.embeddingDim}-dimensional vector. It's literally just array indexing.</div>
          <div className="data-label">With your text:</div>
          <div className="data-stat">Token "{data.state.tokens[0]}" has some ID (let's say 1234). We look up embedding_table[1234] and get [{firstToken?.embedding.slice(0, 3).map(v => v.toFixed(3)).join(', ')}, ...]. That's the vector that will flow through the entire transformer.</div>
          <div className="data-label">What do these dimensions represent?</div>
          <div className="data-stat">Nobody really knows for sure - that's the beauty and mystery of neural networks. Dimension 0 might capture something like "noun-ness", dimension 1 could encode sentiment (positive/negative), dimension 2 might track abstract vs concrete concepts. The model figures this out during training by seeing billions of words in context.</div>
          <div className="data-label">Our output so far:</div>
          <div className="data-stat">A matrix of {data.state.tokens.length} tokens × {data.metadata.embeddingDim} dimensions. Each row is one of your tokens, now represented as a rich semantic vector.</div>
        </div>
      );
    }
  },
  {
    id: 2,
    title: 'Positional Encoding',
    description: 'Here\'s a problem: transformers look at all words at once, in parallel. That\'s fast, but it means they have no idea about word order. "Dog bites man" versus "Man bites dog" would look identical. We fix this by adding positional information directly into the embeddings.',
    highlight: ['encoder-pos'],
    dataFlow: [{ from: 'input-embeddings', to: 'encoder-pos' }],
    showData: (data) => (
      <div className="data-display">
        <div className="data-label">The problem we're solving</div>
        <div className="data-stat">Word order completely changes meaning. "Dog bites man" is very different from "Man bites dog", but without position info, the model can't tell them apart. Unlike older models like LSTMs that processed words one-by-one (and thus knew order naturally), transformers need help.</div>
        <div className="data-label">The solution is elegant</div>
        <div className="data-stat">We add a unique "fingerprint" to each position in the sequence. Position 0 gets one pattern, position 1 gets a different pattern, and so on. Now the model knows not just what each word is, but where it sits in the sentence.</div>
        <div className="data-label">How we generate these fingerprints:</div>
        <div className="formula">PE(pos, 2i) = sin(pos / 10000^(2i/d))</div>
        <div className="formula">PE(pos, 2i+1) = cos(pos / 10000^(2i/d))</div>
        <div className="data-stat">This looks complex, but the idea is simple. For each position (0, 1, 2...) and each dimension of our {data.metadata.embeddingDim}-dimensional embedding, we calculate a value using sine and cosine waves at different frequencies.</div>
        <div className="data-label">Why sine and cosine waves?</div>
        <div className="data-stat">These periodic functions create unique patterns for each position. Low frequencies capture broad position ("beginning vs end of sentence"), while high frequencies capture fine-grained position ("exactly which word"). Plus, the model can learn to detect relative distances - it can figure out that position 5 is two steps from position 3.</div>
        <div className="data-label">Putting it together</div>
        <div className="formula">final_embedding = word_embedding + position_encoding</div>
        <div className="data-stat">We simply add the positional encoding to each word embedding. Your sentence has {data.state.tokens.length} positions (numbered 0 through {data.state.tokens.length - 1}), and each one gets its unique position fingerprint mixed into the embedding.</div>
        <div className="data-stat">Now every token knows both WHAT it is and WHERE it is. That's all we need.</div>
      </div>
    )
  },
  {
    id: 3,
    title: 'Multi-Head Attention (Encoder)',
    description: 'This is where things get interesting. Attention lets each word "look at" every other word in the sentence to gather context. Multiple attention heads run in parallel, each learning to focus on different types of relationships - grammar, semantics, co-references, you name it.',
    highlight: ['encoder-mha'],
    dataFlow: [{ from: 'encoder-pos', to: 'encoder-mha' }],
    showData: (data) => {
      const headDim = Math.floor(data.metadata.embeddingDim / data.metadata.numHeads);
      const encoderAttentions = data.state.encoder[0]?.attention || [];
      return (
        <div className="data-display">
          <div className="data-label">What's happening here</div>
          <div className="data-stat">Think about how you understand language. When you read "it" in a sentence like "The dog barked because it was hungry", you instantly know "it" refers to "the dog". That's attention - figuring out which words connect to which other words.</div>
          <div className="data-label">The attention mechanism uses three matrices:</div>
          <div className="formula">Query (Q): "What am I looking for?"</div>
          <div className="formula">Key (K): "What information do I contain?"</div>
          <div className="formula">Value (V): "Here's my actual content"</div>
          <div className="data-stat">Every token creates its own Q, K, and V by multiplying with learned weight matrices. Then we compare each token's query against every other token's key to figure out who should pay attention to whom.</div>
          <div className="data-label">Computing attention scores</div>
          <div className="formula">attention_scores = Q·K<sup>T</sup> / √{headDim}</div>
          <div className="data-stat">This dot product measures similarity - "how relevant is your key to my query?" We divide by √{headDim} to keep the numbers from getting too large, which would cause problems later.</div>
          <div className="formula">attention_weights = softmax(scores)</div>
          <div className="data-stat">Softmax converts these scores into probabilities that sum to 1.0. Higher scores become higher probabilities.</div>
          <div className="formula">output = attention_weights · V</div>
          <div className="data-stat">Finally, we take a weighted sum of all the Values. If token A has high attention to token B, it pulls in more of B's information.</div>
          <div className="data-label">Why "multi-head"? We're using {data.metadata.numHeads} heads</div>
          <div className="data-stat">Instead of one attention mechanism, we run {data.metadata.numHeads} in parallel. Each head gets to look at {headDim} dimensions (we split the {data.metadata.embeddingDim} dimensions {data.metadata.numHeads} ways). Different heads specialize in different patterns - one might track grammatical relationships like subject-verb, another might focus on semantic similarity, a third might handle pronoun references.</div>
          <div className="data-label">Looking at your input: "{data.state.tokens.join(' ')}"</div>
          {encoderAttentions.slice(0, Math.min(3, data.metadata.numHeads)).map((attention, headIdx) => (
            <div key={headIdx} style={{ marginTop: '6px' }}>
              <div className="data-label">Head {headIdx + 1} attention pattern:</div>
              <div className="attention-preview">
                <div className="matrix-row">
                  {attention.weights[0]?.slice(0, Math.min(5, data.state.tokens.length)).map((w, i) => (
                    <span key={i} className="matrix-cell" title={`Attention to "${data.state.tokens[i]}"`}>
                      {w.toFixed(2)}
                    </span>
                  ))}
                </div>
              </div>
              <div className="data-stat" style={{ fontSize: '10px' }}>How much the first token attends to: {data.state.tokens.slice(0, Math.min(5, data.state.tokens.length)).join(', ')}</div>
            </div>
          ))}
          <div className="data-label" style={{ marginTop: '8px' }}>Combining everything</div>
          <div className="formula">Concatenate all {data.metadata.numHeads} heads → multiply by output weights</div>
          <div className="data-stat">We glue all the head outputs together and pass through one final transformation. Now every token has gathered relevant context from the entire sentence.</div>
        </div>
      );
    }
  },
  {
    id: 4,
    title: 'Add & Norm',
    description: 'Two critical tricks that make deep transformers actually trainable. First, we create a "shortcut" by adding the original input back to the attention output - this gives gradients a highway to flow through during training. Second, we normalize the values to keep everything stable. Without these, the network would either explode or vanish into numerical chaos.',
    highlight: ['encoder-add1'],
    dataFlow: [{ from: 'encoder-mha', to: 'encoder-add1' }],
    showData: (data) => (
      <div className="data-display">
        <div className="data-label">The residual connection</div>
        <div className="formula">output = original_input + attention_output</div>
        <div className="data-stat">Here's the problem this solves. When you stack many layers in a neural network, something nasty happens during training: gradients get progressively smaller as they backpropagate through each layer. Eventually they become so tiny that the early layers barely learn anything. It's called the vanishing gradient problem, and it plagued deep networks for years.</div>
        <div className="data-stat">The solution? Give gradients a shortcut. Instead of forcing them to flow through every transformation, we add the original input directly to the output. Now gradients can flow straight back through this "residual" connection, like a highway bypassing local traffic.</div>
        <div className="data-stat">There's a subtle brilliance here: the model doesn't have to learn the entire transformation from scratch. It just learns what to add or modify. If the best thing to do is nothing, it can simply learn to output zeros, and the residual connection passes the input through unchanged. This makes training dramatically easier.</div>
        <div className="data-label">Layer normalization</div>
        <div className="formula">normalized = (x - μ) / σ</div>
        <div className="formula">output = normalized · γ + β</div>
        <div className="data-stat">Now we normalize. For each token's {data.metadata.embeddingDim}-dimensional vector, we calculate the mean (μ) and standard deviation (σ) across all dimensions. Then we subtract the mean and divide by the standard deviation. This centers the values around zero with unit variance - statisticians call this "standardization".</div>
        <div className="data-stat">Why bother? Training deep networks is like trying to navigate in a coordinate system where the scale keeps changing. One layer might output values around 0.001, the next around 1000. Normalization keeps everything in a consistent range, which makes optimization way more stable and lets you use higher learning rates.</div>
        <div className="data-stat">The γ (gamma) and β (beta) parameters are learned during training. They let the model scale and shift the normalized values if needed. Sometimes you want zero-mean unit-variance, sometimes you don't - let the model decide.</div>
        <div className="data-label">Why this combination works</div>
        <div className="data-stat">Residual connections were the breakthrough that enabled networks like ResNet to train with 100+ layers. LayerNorm was specifically designed for transformers, where it works better than the batch normalization used in CNNs. Together, they're the reason we can stack 12, 24, or even 96 transformer layers and still train them successfully. Without these tricks, you'd be stuck at maybe 2-3 layers before everything collapsed.</div>
      </div>
    )
  },
  {
    id: 5,
    title: 'Feed Forward',
    description: 'After attention mixes information between tokens, each token gets processed through its own little neural network. It\'s wonderfully simple: two linear transformations with a ReLU activation sandwiched between them. Same network, applied to every single token position independently. Think of it as giving each token some private processing time to digest what it just learned from attention.',
    highlight: ['encoder-ff'],
    dataFlow: [{ from: 'encoder-add1', to: 'encoder-ff' }],
    showData: (data) => {
      const ff = data.state.encoder[0]?.feedForward[0];
      const hiddenDim = data.metadata.embeddingDim * 4;
      return (
        <div className="data-display">
          <div className="data-label">A simple two-layer network</div>
          <div className="formula">Layer 1: hidden = ReLU(x·W₁ + b₁)</div>
          <div className="formula">Layer 2: output = hidden·W₂ + b₂</div>
          <div className="data-stat">The architecture is straightforward. We take each token's {data.metadata.embeddingDim}-dimensional vector and multiply it by a weight matrix W₁ to get {hiddenDim} dimensions. That's a 4× expansion - suddenly we have way more room to work with. Then we apply ReLU, which just means "set negative values to zero". Finally, we multiply by W₂ to compress back down to {data.metadata.embeddingDim} dimensions.</div>
          <div className="data-stat">Why expand then compress? Think of the middle layer as a scratchpad. When you're solving a math problem, you write out intermediate steps before arriving at the answer. Same idea here. The model gets {hiddenDim} dimensions to work with internally, which gives it space to compute complex features and combinations. Then it distills everything back to the original dimensionality.</div>
          <div className="data-label">The importance of ReLU</div>
          <div className="data-stat">ReLU (Rectified Linear Unit) is absurdly simple: max(0, x). If the input is positive, pass it through. If it's negative, output zero. That's it. But this simple non-linearity is crucial. Without it, stacking linear layers just gives you... a bigger linear layer. Matrix multiplication is linear, so W₂(W₁x) = (W₂W₁)x - you could collapse the whole thing into a single transformation. ReLU breaks this by introducing non-linearity, letting the network learn complex patterns that no single linear transformation could capture.</div>
          {ff && (
            <>
              <div className="data-label">Sample output from your data</div>
              <div className="vector-preview">
                [{ff.slice(0, 6).map(v => v.toFixed(3)).join(', ')}, ...]
              </div>
            </>
          )}
          <div className="data-label">Independent processing</div>
          <div className="data-stat">Here's something interesting: this feed-forward network is applied to each token completely independently. Token 0 doesn't know what's happening to token 1. This is "position-wise" processing. Attention just mixed information between all the tokens - now each one gets individual processing time. It's like attention was the brainstorming session where everyone talked, and the feed-forward network is everyone going off to think privately about what they learned.</div>
          <div className="data-stat">This alternating pattern - attention to mix information globally, then feed-forward to process locally - is repeated in every transformer layer. It's a remarkably effective design. Attention handles relationships and dependencies, while feed-forward adds computational depth and non-linear modeling capacity.</div>
        </div>
      );
    }
  },
  {
    id: 6,
    title: 'Add & Norm (Final Encoder)',
    description: 'Second residual connection completing the encoder layer',
    highlight: ['encoder-add2'],
    dataFlow: [{ from: 'encoder-ff', to: 'encoder-add2' }],
    showData: (data) => (
      <div className="data-display">
        <div className="data-label">One encoder layer complete</div>
        <div className="data-stat">Another Add & Norm, same as before. We add the residual connection and normalize. By now this should feel familiar - it's the standard stabilization step after each sub-layer.</div>
        <div className="data-label">What just happened in the encoder</div>
        <div className="data-stat">Let's recap the journey. We started with your raw text: "{data.state.tokens.join(' ')}". First we converted those tokens into embeddings - rich {data.metadata.embeddingDim}-dimensional vectors that capture semantic meaning. We mixed in positional information so the model knows word order matters.</div>
        <div className="data-stat">Then came multi-head attention, where each token looked at every other token to gather context. The word "it" might pay attention to "dog" if that's what it refers to. After attention, each token went through a feed-forward network for independent processing. And we stabilized everything with residual connections and layer normalization along the way.</div>
        <div className="data-label">The encoder output</div>
        <div className="data-stat">We now have {data.state.tokens.length} context-aware representations - one for each token. But these aren't the original embeddings anymore. Each vector has been enriched with information from the entire sentence. The representation for "dog" now contains context from "the", "barked", and every other word. Shape: {data.state.tokens.length} tokens × {data.metadata.embeddingDim} dimensions.</div>
        <div className="data-stat">In real transformers, this entire encoder layer (attention → add & norm → feed-forward → add & norm) repeats {data.metadata.numLayers} times. Each layer refines the representations further. By the final layer, the model has a sophisticated understanding of your input text.</div>
        <div className="data-label">What's next: the decoder</div>
        <div className="data-stat">The encoder's job is done. Its output will stay fixed, acting as a memory bank. Now the decoder takes over. While generating output text one token at a time, the decoder will "query" this encoder memory to figure out what parts of the input are relevant. This is how translation works - the French decoder reads the English encoder's memory. This is how summarization works - the summary decoder reads the full article's memory. It's the bridge between input and output.</div>
      </div>
    )
  },
  {
    id: 7,
    title: 'Output Sequence Start',
    description: 'Decoder begins with <START> token or previously generated tokens',
    highlight: ['outputs'],
    dataFlow: [],
    showData: (data) => (
      <div className="data-display">
        <div className="data-label">Entering the decoder</div>
        <div className="data-stat">The encoder processed your input text and built a rich understanding of it. Now the decoder's job is to generate output text, one token at a time. This is called auto-regressive generation - each new token depends on all the tokens that came before it, like writing a sentence word by word.</div>
        <div className="data-label">Two scenarios</div>
        <div className="data-stat">During training, we already know what the output should be. So we feed the target sentence into the decoder (shifted right with a &lt;START&gt; token prepended). The decoder learns to predict each next token given all the previous ones.</div>
        <div className="data-stat">During inference - when actually using the model - we don't have the target sentence. We start with just &lt;START&gt;, generate the first token, add it to the sequence, generate the second token, and keep going until we produce &lt;END&gt;. For your input "{data.state.tokens.join(' ')}", the decoder might generate: &lt;START&gt; → {data.state.tokens[0]} → {data.state.tokens[1]} → ... → &lt;END&gt;.</div>
        <div className="data-label">Starting sequence</div>
        <div className="token-list">
          <span className="token-badge">&lt;START&gt;</span>
          {data.state.tokens.slice(0, 3).map((token, i) => (
            <span key={i} className="token-badge">{token}</span>
          ))}
        </div>
        <div className="data-label">The key difference</div>
        <div className="data-stat">The encoder sees the entire input sentence at once and processes it in parallel. Every token can attend to every other token freely. But the decoder has to maintain causality - when generating token at position i, it can only see positions 0 through i-1. It can't peek into the future. Otherwise during training, the model would just "cheat" by copying the answer from the target sequence.</div>
        <div className="data-stat">Coming up: we'll embed these output tokens, add positional encodings (same as encoder), and then things get interesting with masked attention that enforces this causality constraint.</div>
      </div>
    )
  },
  {
    id: 8,
    title: 'Output Embeddings',
    description: 'Output tokens embedded into vector space',
    highlight: ['output-embeddings'],
    dataFlow: [{ from: 'outputs', to: 'output-embeddings' }],
    showData: (data) => (
      <div className="data-display">
        <div className="data-label">Same embedding process</div>
        <div className="data-stat">Just like in the encoder, we take the decoder's tokens (either from the target sequence during training, or generated so far during inference) and look them up in the embedding table. Each discrete token ID becomes a continuous {data.metadata.embeddingDim}-dimensional vector.</div>
        <div className="data-label">Shared embeddings</div>
        <div className="data-stat">Here's something clever: the encoder and decoder typically share the same embedding table. The word "cat" gets the exact same {data.metadata.embeddingDim}-dimensional vector whether it appears in the input or output. This makes sense - it's the same vocabulary, often the same language. Sharing embeddings saves memory (only one table of {data.metadata.vocabSize} embeddings to store) and helps the model learn better representations since it sees each word in both contexts.</div>
        <div className="data-stat">For sequence-to-sequence tasks where input and output are different languages (like English to French translation), you might use separate embedding tables. But for tasks where both sides speak the same language, sharing is common and effective.</div>
        <div className="data-label">Next up</div>
        <div className="data-stat">We'll add positional encodings - the decoder needs to know position information just like the encoder did. Then comes masked self-attention, where we enforce the causality constraint so the decoder can't cheat by looking at future tokens.</div>
      </div>
    )
  },
  {
    id: 9,
    title: 'Positional Encoding (Decoder)',
    description: 'Position information added to decoder embeddings',
    highlight: ['decoder-pos'],
    dataFlow: [{ from: 'output-embeddings', to: 'decoder-pos' }],
    showData: (data) => (
      <div className="data-display">
        <div className="data-label">Adding position information</div>
        <div className="data-stat">We use the exact same sine/cosine positional encoding as the encoder. Each position (0, 1, 2, ...) gets its unique fingerprint mixed into the embedding. Word order matters just as much in the output as it did in the input - "the cat" and "cat the" mean very different things.</div>
        <div className="data-stat">During generation, positional encodings tell the model "where we are" in the output sequence. When generating the first word, we're at position 0. Second word, position 1. And so on. This position information helps the model understand the structure of what it's building.</div>
        <div className="data-label">Ready for attention</div>
        <div className="data-stat">We now have decoder tokens as rich embeddings with positional information baked in. Next comes masked self-attention, where the decoder looks at its own previous outputs (without peeking ahead). After that, cross-attention will let the decoder read the encoder's memory to ground its generation in the input.</div>
      </div>
    )
  },
  {
    id: 10,
    title: 'Masked Multi-Head Attention',
    description: 'Self-attention in the decoder, but with a crucial restriction: each position can only attend to previous positions, never future ones. We enforce this with a mask that sets future attention scores to negative infinity. This maintains the auto-regressive property - when predicting the next token, you only get to see what came before, never what comes after.',
    highlight: ['decoder-masked'],
    dataFlow: [{ from: 'decoder-pos', to: 'decoder-masked' }],
    showData: (data) => {
      const attention = data.state.decoder[0]?.attention[0];
      return (
        <div className="data-display">
          <div className="data-label">The masking mechanism</div>
          <div className="formula">scores = QK^T / √d_k</div>
          <div className="formula">scores[i,j] = -∞ if j &gt; i (MASK)</div>
          <div className="formula">attention = softmax(scores) · V</div>
          <div className="data-stat">We compute attention scores normally - dot products between queries and keys. But before applying softmax, we mask out the future. For any position i trying to attend to position j where j &gt; i (a future position), we set that attention score to negative infinity.</div>
          <div className="data-stat">Why negative infinity? When you take softmax of -∞, you get exactly 0. So future positions end up with zero attention weight - completely ignored, as if they don't exist yet. Mathematically clean, and it works perfectly.</div>
          <div className="data-label">The causal mask pattern</div>
          <div className="mask-visual" style={{ fontFamily: 'monospace', fontSize: '14px', lineHeight: '1.6' }}>
            Position 0: [✓ ✗ ✗ ✗] ← can only see itself<br/>
            Position 1: [✓ ✓ ✗ ✗] ← sees pos 0,1<br/>
            Position 2: [✓ ✓ ✓ ✗] ← sees pos 0,1,2<br/>
            Position 3: [✓ ✓ ✓ ✓] ← sees all previous
          </div>
          <div className="data-stat">This creates a lower-triangular attention matrix. Position 0 only sees itself (the &lt;START&gt; token). Position 1 sees positions 0 and 1. Position 2 sees 0, 1, and 2. Each position gets to attend to all previous positions plus itself, but nothing beyond.</div>
          <div className="data-stat">This is crucial for training. If we didn't mask, the decoder could just look ahead and copy the answer from the target sequence. It would learn nothing. Masking forces the model to truly learn to generate the next token based only on what it's seen so far - the exact task it needs to do during inference.</div>
        </div>
      );
    }
  },
  {
    id: 11,
    title: 'Add & Norm (Decoder)',
    description: 'Residual connection after masked self-attention',
    highlight: ['decoder-add1'],
    dataFlow: [{ from: 'decoder-masked', to: 'decoder-add1' }],
    showData: () => (
      <div className="data-display">
        <div className="data-label">Stabilization after masked attention</div>
        <div className="formula">output = LayerNorm(x + MaskedAttention(x))</div>
        <div className="data-stat">Same deal as before: residual connection plus layer normalization. Add the original input back to prevent vanishing gradients, then normalize to keep values stable. You've seen this pattern multiple times now - it appears after every sub-layer in the transformer.</div>
        <div className="data-label">Progress check</div>
        <div className="data-stat">The decoder has embedded its output tokens, added positional encodings, and run masked self-attention so each position can gather context from previous positions. Now we're about to do something really interesting.</div>
        <div className="data-label">The magic moment approaches</div>
        <div className="data-stat">Next comes cross-attention - this is where the decoder actually reads the encoder's memory. Up until now, the decoder has only looked at its own output sequence. Cross-attention is the bridge that connects the input (encoder) to the output (decoder). In a translation model, this is where the French decoder reads the English encoder's understanding. In a summarization model, where the summary decoder reads the full article. It's the mechanism that grounds generation in the input.</div>
      </div>
    )
  },
  {
    id: 12,
    title: 'Cross-Attention',
    description: 'Decoder attends to encoder output: Q from decoder, K & V from encoder',
    highlight: ['decoder-cross', 'encoder-add2'],
    dataFlow: [
      { from: 'decoder-add1', to: 'decoder-cross' },
      { from: 'encoder-add2', to: 'decoder-cross' }
    ],
    showData: (data) => {
      const decoderAttentions = data.state.decoder[0]?.attention || [];
      return (
        <div className="data-display">
          <div className="data-label">The bridge between input and output</div>
          <div className="data-stat">This is the most important step in the entire transformer architecture. Cross-attention is how the decoder actually reads and uses the input. Everything before this was processing the input (encoder) or processing past outputs (masked self-attention). Now we connect them.</div>
          <div className="data-label">How cross-attention differs from self-attention</div>
          <div className="data-stat">In self-attention, tokens within the same sequence attend to each other. Queries, keys, and values all come from the same place. Cross-attention is different: queries come from the decoder ("what am I looking for?"), while keys and values come from the encoder ("here's what the input contains").</div>
          <div className="formula">Q = decoder_state · W_Q</div>
          <div className="formula">K = encoder_output · W_K</div>
          <div className="formula">V = encoder_output · W_V</div>
          <div className="formula">CrossAttn = softmax(Q·K<sup>T</sup>/√d)·V</div>
          <div className="data-stat">The decoder creates queries based on what it's trying to generate. These queries are matched against keys from the encoder to figure out which parts of the input are relevant right now. Then we pull the corresponding values - the actual content from those relevant input positions.</div>
          <div className="data-label">What this looks like with your data</div>
          <div className="data-stat">Your input is: "{data.state.tokens.join(' ')}". As the decoder generates output word by word, cross-attention lets it ask "which input words matter for what I'm generating now?" When generating the first output token, it might focus heavily on "{data.state.tokens[0]}". For the second token, maybe "{data.state.tokens[1] || data.state.tokens[0]}". Different output positions attend to different parts of the input.</div>
          {decoderAttentions.length > 0 && (
            <>
              <div className="data-label">Attention weights: decoder → encoder</div>
              {decoderAttentions.slice(0, 2).map((attention, headIdx) => (
                <div key={headIdx} style={{ marginTop: '6px' }}>
                  <div className="data-label">Head {headIdx + 1}</div>
                  <div className="attention-preview">
                    <div className="matrix-row">
                      {attention.weights[0]?.slice(0, Math.min(6, data.state.tokens.length)).map((w, i) => (
                        <span key={i} className="matrix-cell" title={`Attention to input token ${i}`}>
                          {w.toFixed(3)}
                        </span>
                      ))}
                    </div>
                  </div>
                </div>
              ))}
            </>
          )}
          <div className="data-label">Why this matters</div>
          <div className="data-stat">Cross-attention is what prevents hallucination and keeps the output grounded in the input. When translating, the model can't just make things up - it has to attend to the source text. When summarizing, it has to reference the actual article. The decoder constantly asks "what does the input say?" via cross-attention and bases its generation on the answer. It's the fundamental mechanism that makes transformers so effective for sequence-to-sequence tasks.</div>
        </div>
      );
    }
  },
  {
    id: 13,
    title: 'Add & Norm (Decoder)',
    description: 'Residual connection after cross-attention',
    highlight: ['decoder-add2'],
    dataFlow: [{ from: 'decoder-cross', to: 'decoder-add2' }],
    showData: (data) => (
      <div className="data-display">
        <div className="data-label">Combining two contexts</div>
        <div className="formula">output = LayerNorm(x + CrossAttention(x, encoder))</div>
        <div className="data-stat">Another residual connection and layer normalization. At this point, the decoder state is rich with information from two sources: masked self-attention gave it context from past output tokens, and cross-attention gave it context from the entire input sequence.</div>
        <div className="data-stat">For your input "{data.state.tokens.join(' ')}", imagine the decoder is generating position N. It knows what it's generated so far (positions 0 through N-1, thanks to masked attention). And it knows what the input says (all input tokens, thanks to cross-attention). Both contexts are now fused together in the decoder's representation.</div>
        <div className="data-label">Almost done with this decoder layer</div>
        <div className="data-stat">Next comes a feed-forward network for independent processing of each position, followed by one more Add & Norm. Then we'll project to vocabulary size and pick the actual output token. The decoder layer structure mirrors the encoder: attention → add & norm → feed-forward → add & norm. The difference is the decoder has two attention steps (masked self, then cross) while the encoder has just one.</div>
      </div>
    )
  },
  {
    id: 14,
    title: 'Feed Forward (Decoder)',
    description: 'Position-wise feed-forward network',
    highlight: ['decoder-ff'],
    dataFlow: [{ from: 'decoder-add2', to: 'decoder-ff' }],
    showData: (data) => (
      <div className="data-display">
        <div className="data-label">Position-wise processing</div>
        <div className="formula">FFN(x) = max(0, x·W₁ + b₁)·W₂ + b₂</div>
        <div className="data-stat">Same feed-forward network as in the encoder. Expand from {data.metadata.embeddingDim} to {data.metadata.embeddingDim * 4} dimensions with ReLU activation, then compress back down. Applied independently to each token position - no interaction between positions at this stage.</div>
        <div className="data-stat">The attention mechanisms (both masked self-attention and cross-attention) mixed information globally. Now each token gets individual processing to transform and refine what it's learned. This alternating pattern - global mixing via attention, then local processing via feed-forward - repeats throughout both the encoder and decoder.</div>
        <div className="data-label">The decoder layer structure</div>
        <div className="data-stat">Each decoder layer follows the same pattern: masked self-attention (look at past outputs) → add & norm → cross-attention (look at input) → add & norm → feed-forward (independent processing) → add & norm. This whole block repeats {data.metadata.numLayers} times in a real transformer, with each layer adding depth and refinement to the generation process.</div>
      </div>
    )
  },
  {
    id: 15,
    title: 'Add & Norm (Final Decoder)',
    description: 'Final residual connection completing decoder layer',
    highlight: ['decoder-add3'],
    dataFlow: [{ from: 'decoder-ff', to: 'decoder-add3' }],
    showData: (data) => (
      <div className="data-display">
        <div className="data-label">One decoder layer complete</div>
        <div className="formula">output = LayerNorm(x + FeedForward(x))</div>
        <div className="data-stat">Final Add & Norm for this layer. We've now gone through the full decoder layer: masked self-attention to incorporate past outputs, cross-attention to read the input, feed-forward for complex transformations, with residual connections and normalization stabilizing everything along the way.</div>
        <div className="data-stat">In production transformers, this entire layer structure repeats {data.metadata.numLayers} times. The original "Attention is All You Need" paper used 6 layers. GPT-3 uses 96. Each additional layer refines the representations and improves the model's ability to generate coherent, relevant output.</div>
        <div className="data-label">What we have now</div>
        <div className="data-stat">After all these transformations, we have rich {data.metadata.embeddingDim}-dimensional vectors for each position in the output sequence. These vectors encode context from both past output tokens and the entire input sequence. Shape: {data.state.tokens.length} positions × {data.metadata.embeddingDim} dimensions.</div>
        <div className="data-label">Final steps</div>
        <div className="data-stat">Now we need to convert these continuous {data.metadata.embeddingDim}-dimensional vectors into actual discrete tokens. We'll project to vocabulary size ({data.metadata.vocabSize} dimensions), getting a score for every possible next token. Then softmax converts scores to probabilities, and we pick the most likely one (or sample from the distribution). Almost there.</div>
      </div>
    )
  },
  {
    id: 16,
    title: 'Linear Projection',
    description: 'Projects decoder output to vocabulary size: W · x + b',
    highlight: ['linear'],
    dataFlow: [{ from: 'decoder-add3', to: 'linear' }],
    showData: (data) => (
      <div className="data-display">
        <div className="data-label">Projecting to vocabulary space</div>
        <div className="formula">logits = decoder_output · W_vocab + bias</div>
        <div className="data-stat">We take the decoder's {data.metadata.embeddingDim}-dimensional output and multiply it by a huge weight matrix with shape [{data.metadata.embeddingDim} × {data.metadata.vocabSize}]. The result: {data.metadata.vocabSize} numbers, one score for every possible token in the vocabulary. These raw scores are called "logits".</div>
        <div className="data-stat">Logits aren't probabilities - they can be negative, they can be greater than 1, they don't sum to 1. They're just scores. Higher numbers mean the model thinks that token is more likely to come next. Maybe "{data.state.tokens[0]}" gets a score of 4.2, while "{data.state.tokens[1] || 'another'}" gets 5.7. We'll convert these to proper probabilities in the next step.</div>
        <div className="data-label">How the model learned this</div>
        <div className="data-stat">During training, the weight matrix W_vocab learned to map decoder states to appropriate next tokens. Each column corresponds to one vocabulary token. When you multiply the decoder output by W_vocab, you're essentially computing {data.metadata.vocabSize} dot products - measuring how well the decoder state matches the pattern for each possible next word.</div>
        <div className="data-stat">If the decoder has built a representation that screams "the next word should be '{data.state.tokens[0]}'", then the dot product with that token's column will be high. If the representation doesn't match some unrelated word, that dot product will be low or negative. It's pattern matching at scale.</div>
      </div>
    )
  },
  {
    id: 17,
    title: 'Softmax & Output',
    description: 'Convert logits to probabilities and select next token',
    highlight: ['softmax'],
    dataFlow: [{ from: 'linear', to: 'softmax' }],
    showData: (data) => (
      <div className="data-display">
        <div className="data-label">Converting logits to probabilities</div>
        <div className="formula">P(token_i) = exp(logit_i) / Σⱼ exp(logit_j)</div>
        <div className="data-stat">Softmax turns raw scores into a proper probability distribution. We exponentiate each logit, then divide by the sum of all exponentiated logits. This ensures all probabilities are positive and sum to exactly 1.0. Higher logits become higher probabilities - if "{data.state.tokens[2] || 'word3'}" had a logit of 5.7 and "{data.state.tokens[0]}" had 4.2, softmax might give them probabilities of 0.73 and 0.22 respectively.</div>
        <div className="data-label">Example with your input tokens</div>
        <div className="token-list">
          <div className="data-label">Top predictions</div>
          {data.state.tokens.slice(0, 3).map((token, i) => (
            <span key={i} className="token-badge">
              {token} (p={(0.9 - i * 0.15).toFixed(3)})
            </span>
          ))}
        </div>
        <div className="data-label">How to pick the next token</div>
        <div className="data-stat">Greedy decoding just picks the highest probability token every time. Simple and deterministic, but can be repetitive and boring. Sampling adds randomness - pick tokens according to their probability distribution. A token with 0.3 probability gets chosen 30% of the time. This produces more diverse, creative output.</div>
        <div className="data-stat">Temperature scaling controls randomness. Before softmax, divide all logits by temperature T. Low temperature (T=0.1) sharpens the distribution - the top choice becomes even more dominant, giving conservative, predictable output. High temperature (T=1.5) flattens the distribution, giving more weight to less likely tokens and producing creative, sometimes wild results.</div>
        <div className="data-stat">Top-k sampling restricts the pool to the k most likely tokens (say, top 40). Top-p (nucleus) sampling includes tokens until their cumulative probability reaches p (say, 0.9). Both prevent the model from occasionally choosing bizarre low-probability words that would derail the output.</div>
        <div className="data-label">The generation loop</div>
        <div className="data-stat">We've generated one token. Now we add it to the output sequence and feed the whole thing back into the decoder. Generate the next token. Add it. Feed back. Repeat until we generate &lt;END&gt;, or hit a maximum length, or the user stops us. This is auto-regressive generation - each step depends on all previous steps.</div>
        <div className="data-label">Journey complete</div>
        <div className="data-stat">The encoder processed "{data.state.tokens.join(' ')}" into a deep semantic understanding. The decoder generated output one token at a time, using masked self-attention to track what it's generated, cross-attention to stay grounded in the input, and feed-forward layers to process complex patterns. Residual connections and normalization kept everything trainable. And here we are: from raw text to rich representations to generated output. That's the transformer architecture.</div>
      </div>
    )
  }
];

export const Walkthrough: React.FC = () => {
  const [currentStep, setCurrentStep] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [text, setText] = useState('The transformer learns patterns');
  const [data, setData] = useState<VisualizationResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [theme, setTheme] = useState<'dark' | 'light'>('dark');
  const walkthroughRef = React.useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (text.trim()) {
      loadData();
    }
  }, []);

  const loadData = async () => {
    setLoading(true);
    try {
      const result = await visualizeTransformer(text, 6, 8, 512);
      setData(result);
    } catch (error) {
      console.error('Failed to load data:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleVisualize = () => {
    setCurrentStep(0);
    loadData();
  };

  useEffect(() => {
    if (isPlaying && data) {
      const timer = setTimeout(() => {
        if (currentStep < steps.length - 1) {
          setCurrentStep(currentStep + 1);
        } else {
          setIsPlaying(false);
        }
      }, 3500);
      return () => clearTimeout(timer);
    }
  }, [isPlaying, currentStep, data]);

  const handleNext = () => {
    if (currentStep < steps.length - 1) {
      setCurrentStep(currentStep + 1);
    }
  };

  const handlePrev = () => {
    if (currentStep > 0) {
      setCurrentStep(currentStep - 1);
    }
  };

  const handlePlay = () => {
    setIsPlaying(!isPlaying);
  };

  const handleFullscreen = () => {
    if (!isFullscreen) {
      if (walkthroughRef.current?.requestFullscreen) {
        walkthroughRef.current.requestFullscreen();
      }
    } else {
      if (document.exitFullscreen) {
        document.exitFullscreen();
      }
    }
  };

  const handleThemeToggle = () => {
    setTheme(theme === 'dark' ? 'light' : 'dark');
  };

  useEffect(() => {
    const handleFullscreenChange = () => {
      setIsFullscreen(!!document.fullscreenElement);
    };
    document.addEventListener('fullscreenchange', handleFullscreenChange);
    return () => document.removeEventListener('fullscreenchange', handleFullscreenChange);
  }, []);

  const step = steps[currentStep];

  return (
    <div className={`walkthrough ${theme}`} ref={walkthroughRef}>
      <motion.div
        className="walkthrough-header glass"
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
      >
        <div className="header-row">
          <input
            type="text"
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="Enter text to analyze..."
            className="text-input"
            disabled={loading}
          />
          <button onClick={handleVisualize} disabled={loading} className="glass-button">
            {loading ? 'Processing...' : 'Analyze'}
          </button>
          <button 
            onClick={handleThemeToggle} 
            className="glass-button theme-btn"
            title={theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'}
          >
            {theme === 'dark' ? '☀️' : '🌙'}
          </button>
          <div className="step-indicator">
            Step {currentStep + 1}/{steps.length}
          </div>
        </div>
      </motion.div>

      <button 
        onClick={handleFullscreen} 
        className="glass-button floating-fullscreen-btn"
        title={isFullscreen ? 'Exit fullscreen (Esc)' : 'Enter fullscreen'}
      >
        {isFullscreen ? '✕ Exit' : '⛶ Fullscreen'}
      </button>

      {data && (
        <>
          <div className="walkthrough-content">
            <div className="architecture-container">
              <TransformerDiagram highlight={step.highlight} dataFlow={step.dataFlow} theme={theme} />
            </div>

            <AnimatePresence mode="wait">
              <motion.div
                key={currentStep}
                className="step-info glass"
                initial={{ opacity: 0, x: 50 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -50 }}
                transition={{ duration: 0.4 }}
              >
                <h3>{step.title}</h3>
                
                <div className="calculations">
                  {step.showData(data)}
                </div>
                
                <div className="progress-bar">
                  <motion.div
                    className="progress-fill"
                    initial={{ width: 0 }}
                    animate={{ width: `${((currentStep + 1) / steps.length) * 100}%` }}
                    transition={{ duration: 0.5 }}
                  />
                </div>
              </motion.div>
            </AnimatePresence>
          </div>

          <div className="controls glass">
            <button onClick={handlePrev} disabled={currentStep === 0} className="glass-button">
              ← Previous
            </button>
            <button onClick={handlePlay} className="glass-button play-btn">
              {isPlaying ? '⏸ Pause' : '▶ Play'}
            </button>
            <button onClick={handleNext} disabled={currentStep === steps.length - 1} className="glass-button">
              Next →
            </button>
          </div>
        </>
      )}
    </div>
  );
};

interface DiagramProps {
  highlight: string[];
  dataFlow: { from: string; to: string }[];
  theme: 'dark' | 'light';
}

const TransformerDiagram: React.FC<DiagramProps> = ({ highlight, dataFlow, theme }) => {
  const isHighlighted = (id: string) => highlight.includes(id);
  const isDark = theme === 'dark';
  const textColor = isDark ? '#fff' : '#1a1a1a';
  const strokeColor = isDark ? 'rgba(255,255,255,0.1)' : 'rgba(0,0,0,0.2)';
  const highlightStroke = isDark ? '#fff' : '#000';

  return (
    <svg viewBox="0 0 700 850" className="transformer-diagram" preserveAspectRatio="xMidYMid meet">
      <defs>
        <linearGradient id="encoderGrad" x1="0%" y1="0%" x2="0%" y2="100%">
          <stop offset="0%" stopColor="#2d5f3f" stopOpacity="0.3" />
          <stop offset="100%" stopColor="#1a3a28" stopOpacity="0.3" />
        </linearGradient>
        <linearGradient id="decoderGrad" x1="0%" y1="0%" x2="0%" y2="100%">
          <stop offset="0%" stopColor="#3d5a7f" stopOpacity="0.3" />
          <stop offset="100%" stopColor="#1f2f47" stopOpacity="0.3" />
        </linearGradient>
        <linearGradient id="arrowGradient" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stopColor="#0ea5e9" stopOpacity="0.4" />
          <stop offset="50%" stopColor="#06b6d4" stopOpacity="0.9" />
          <stop offset="100%" stopColor="#0ea5e9" stopOpacity="0.4" />
        </linearGradient>
        <marker id="arrowhead" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto">
          <polygon points="0 0, 10 3, 0 6" fill="#06b6d4" />
        </marker>
        <filter id="glow">
          <feGaussianBlur stdDeviation="4" result="coloredBlur" />
          <feMerge>
            <feMergeNode in="coloredBlur" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>
      </defs>

      {/* Inputs */}
      <Component id="inputs" x={70} y={685} width={180} height={60} label="Inputs" 
        highlighted={isHighlighted('inputs')} color="#FDB750" />

      {/* Input Embeddings */}
      <Component id="input-embeddings" x={70} y={605} width={180} height={60} 
        label="Input Embeddings" highlighted={isHighlighted('input-embeddings')} color="#FDB750" />

      {/* Encoder Positional */}
      <Component id="encoder-pos" x={5} y={525} width={180} height={60} 
        label="Positional Encoding" highlighted={isHighlighted('encoder-pos')} color="#FFFFFF" />

      {/* Encoder Block */}
      <rect x={0} y={165} width={240} height={340} rx="16" fill="url(#encoderGrad)" 
        stroke={highlight.some(h => h.startsWith('encoder')) ? '#4ade80' : strokeColor} 
        strokeWidth="2" />
      <text x={120} y={150} textAnchor="middle" fill={isDark ? '#4ade80' : '#16a34a'} fontSize="16" fontWeight="600">
        ENCODER
      </text>

      {/* Encoder Components */}
      <Component id="encoder-mha" x={15} y={440} width={210} height={60} 
        label="Multi-Head Attention" highlighted={isHighlighted('encoder-mha')} color="#FFD27F" />
      <Component id="encoder-add1" x={30} y={370} width={180} height={55} 
        label="Add & Norm" highlighted={isHighlighted('encoder-add1')} color="#D4A5FF" />
      <Component id="encoder-ff" x={15} y={300} width={210} height={60} 
        label="Feed Forward" highlighted={isHighlighted('encoder-ff')} color="#FFD27F" />
      <Component id="encoder-add2" x={30} y={230} width={180} height={55} 
        label="Add & Norm" highlighted={isHighlighted('encoder-add2')} color="#D4A5FF" />

      {/* Cross-attention arrow indicator */}
      {highlight.includes('decoder-cross') && highlight.includes('encoder-add2') && (
        <motion.path
          d="M 240 257 L 455 330"
          stroke="#06b6d4"
          strokeWidth="3"
          fill="none"
          strokeDasharray="6,3"
          initial={{ pathLength: 0, opacity: 0 }}
          animate={{ pathLength: 1, opacity: 0.7 }}
          transition={{ duration: 1.2, ease: 'easeInOut' }}
          markerEnd="url(#arrowhead)"
        />
      )}

      {/* Outputs */}
      <Component id="outputs" x={450} y={685} width={180} height={60} 
        label="Outputs" highlighted={isHighlighted('outputs')} color="#FDB750" />

      {/* Output Embeddings */}
      <Component id="output-embeddings" x={450} y={605} width={180} height={60} 
        label="Output Embeddings" highlighted={isHighlighted('output-embeddings')} color="#FDB750" />

      {/* Decoder Positional */}
      <Component id="decoder-pos" x={450} y={525} width={180} height={60} 
        label="Positional Encoding" highlighted={isHighlighted('decoder-pos')} color="#FFFFFF" />

      {/* Decoder Block */}
      <rect x={430} y={30} width={260} height={475} rx="16" fill="url(#decoderGrad)" 
        stroke={highlight.some(h => h.startsWith('decoder')) ? '#60a5fa' : strokeColor} 
        strokeWidth="2" />
      <text x={705} y={267} textAnchor="middle" fill={isDark ? '#60a5fa' : '#2563eb'} fontSize="16" fontWeight="600"
        transform="rotate(90, 705, 267)">
        DECODER
      </text>

      {/* Decoder Components */}
      <Component id="decoder-masked" x={445} y={440} width={230} height={60} 
        label="Masked Multi-Head Attention" highlighted={isHighlighted('decoder-masked')} color="#FFD27F" />
      <Component id="decoder-add1" x={465} y={370} width={190} height={55} 
        label="Add & Norm" highlighted={isHighlighted('decoder-add1')} color="#D4A5FF" />
      <Component id="decoder-cross" x={445} y={300} width={230} height={60} 
        label="Multi-Head Attention" highlighted={isHighlighted('decoder-cross')} color="#FFD27F" />
      <Component id="decoder-add2" x={465} y={230} width={190} height={55} 
        label="Add & Norm" highlighted={isHighlighted('decoder-add2')} color="#D4A5FF" />
      <Component id="decoder-ff" x={445} y={160} width={230} height={60} 
        label="Feed Forward" highlighted={isHighlighted('decoder-ff')} color="#FFD27F" />
      <Component id="decoder-add3" x={465} y={90} width={190} height={55} 
        label="Add & Norm" highlighted={isHighlighted('decoder-add3')} color="#D4A5FF" />

      {/* Output Layers */}
      <Component id="linear" x={490} y={45} width={140} height={35} 
        label="Linear" highlighted={isHighlighted('linear')} color="#FFFFFF" />
      <Component id="softmax" x={485} y={5} width={150} height={30} 
        label="Softmax" highlighted={isHighlighted('softmax')} color="#FFFFFF" />

      {/* Final Output */}
      <text x={560} y={-5} textAnchor="middle" fill={textColor} fontSize="10" fontWeight="500">
        Output Probabilities
      </text>

      {/* Arrows */}
      <Arrows dataFlow={dataFlow} />
    </svg>
  );
};

interface ComponentProps {
  id: string;
  x: number;
  y: number;
  width: number;
  height: number;
  label: string;
  highlighted: boolean;
  color: string;
}

const Component: React.FC<ComponentProps> = ({ id, x, y, width, height, label, highlighted, color }) => {
  // Darken text for light backgrounds, lighten for dark backgrounds
  const textColor = highlighted ? '#000' : '#666';
  
  return (
    <motion.g
      initial={{ opacity: 0, scale: 0.8 }}
      animate={{ 
        opacity: highlighted ? 1 : 0.4,
        scale: highlighted ? 1.05 : 1
      }}
      transition={{ duration: 0.5 }}
    >
      <rect
        x={x}
        y={y}
        width={width}
        height={height}
        rx="12"
        fill={highlighted ? color : `${color}40`}
        stroke={highlighted ? '#fff' : 'rgba(255,255,255,0.2)'}
        strokeWidth={highlighted ? '3' : '1'}
        filter={highlighted ? 'url(#glow)' : ''}
      />
      <text
        x={x + width / 2}
        y={y + height / 2 + 5}
        textAnchor="middle"
        fill={highlighted ? '#000' : '#888'}
        fontSize="14"
        fontWeight={highlighted ? '600' : '500'}
      >
        {label}
      </text>
    </motion.g>
  );
};

const Arrows: React.FC<{ dataFlow: { from: string; to: string }[] }> = ({ dataFlow }) => {
  const positions: Record<string, { x: number; y: number }> = {
    'inputs': { x: 160, y: 715 },
    'input-embeddings': { x: 160, y: 635 },
    'encoder-pos': { x: 95, y: 555 },
    'encoder-mha': { x: 120, y: 470 },
    'encoder-add1': { x: 120, y: 397 },
    'encoder-ff': { x: 120, y: 330 },
    'encoder-add2': { x: 120, y: 257 },
    'outputs': { x: 540, y: 715 },
    'output-embeddings': { x: 540, y: 635 },
    'decoder-pos': { x: 540, y: 555 },
    'decoder-masked': { x: 560, y: 470 },
    'decoder-add1': { x: 560, y: 397 },
    'decoder-cross': { x: 560, y: 330 },
    'decoder-add2': { x: 560, y: 257 },
    'decoder-ff': { x: 560, y: 190 },
    'decoder-add3': { x: 560, y: 117 },
    'linear': { x: 560, y: 62 },
    'softmax': { x: 560, y: 20 }
  };

  const createPath = (from: { x: number; y: number }, to: { x: number; y: number }): string => {
    const dx = to.x - from.x;
    const dy = to.y - from.y;
    const absDx = Math.abs(dx);
    const absDy = Math.abs(dy);

    if (absDx < 10) {
      return `M ${from.x} ${from.y} L ${to.x} ${to.y}`;
    }

    if (dx > 100) {
      const midX = from.x + dx * 0.5;
      return `M ${from.x} ${from.y} C ${midX} ${from.y}, ${midX} ${to.y}, ${to.x} ${to.y}`;
    }

    const controlOffset = Math.min(absDy * 0.3, 40);
    return `M ${from.x} ${from.y} C ${from.x} ${from.y - controlOffset}, ${to.x} ${to.y + controlOffset}, ${to.x} ${to.y}`;
  };

  return (
    <g>
      {dataFlow.map((flow, idx) => {
        const from = positions[flow.from];
        const to = positions[flow.to];
        if (!from || !to) return null;

        const pathD = createPath(from, to);

        return (
          <g key={idx}>
            <motion.path
              d={pathD}
              fill="none"
              stroke="url(#arrowGradient)"
              strokeWidth="3"
              strokeLinecap="round"
              initial={{ pathLength: 0, opacity: 0 }}
              animate={{ pathLength: 1, opacity: 1 }}
              transition={{ duration: 1, ease: 'easeInOut' }}
            />
            <defs>
              <path id={`path-${idx}`} d={pathD} />
            </defs>
          </g>
        );
      })}
    </g>
  );
};
