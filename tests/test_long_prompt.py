"""CPU-only tests for chunk boundaries and SDXL embedding assembly."""
import types
import unittest
from colab import studio


class Tokenizer:
    model_max_length = 77
    bos_token_id = 1000
    eos_token_id = 1001
    pad_token_id = 0

    def __call__(self, text, **kwargs):
        assert kwargs == {'add_special_tokens': False, 'truncation': False}
        return types.SimpleNamespace(input_ids=list(range(1, len(text.split()) + 1)))


class LongPromptTests(unittest.TestCase):
    def test_boundaries_no_truncation_and_break(self):
        tokenizer = Tokenizer()
        for count, expected in [(0, 1), (75, 1), (76, 2), (150, 2), (199, 3), (200, 3)]:
            chunks = studio.clip_token_chunks(tokenizer, ' '.join(['word'] * count))
            self.assertEqual(len(chunks), expected)
            self.assertEqual(sum(map(len, chunks)), count)
        self.assertEqual(len(studio.clip_token_chunks(tokenizer, 'first BREAK second')), 2)
        with self.assertRaisesRegex(ValueError, '200 token'):
            studio.clip_token_chunks(tokenizer, ' '.join(['word'] * 201))
        with self.assertRaises(ValueError):
            studio.clip_token_chunks(tokenizer, ' BREAK ' * 3)

    def test_token_limit_counts_across_breaks(self):
        tokenizer = Tokenizer()
        prompt = 'word ' * 75 + 'BREAK ' + 'word ' * 75 + 'BREAK ' + 'word ' * 50
        self.assertEqual([len(c) for c in studio.clip_token_chunks(tokenizer, prompt)], [75, 75, 50])
        with self.assertRaisesRegex(ValueError, '200 token'):
            studio.clip_token_chunks(tokenizer, prompt + 'word')

    def test_embeddings_align_both_encoders_and_both_prompts(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest('NumPy needed for lightweight tensor double')

        class Tensor:
            def __init__(self, data): self.data = np.asarray(data)
            def to(self, **kwargs): return self
            def mean(self, dim): return Tensor(self.data.mean(axis=dim))

        torch = types.SimpleNamespace(
            long='long', tensor=lambda data, **kw: Tensor(data),
            cat=lambda items, dim: Tensor(np.concatenate([x.data for x in items], axis=dim)),
            stack=lambda items: Tensor(np.stack([x.data for x in items])),
        )

        class Encoder:
            dtype = 'float32'
            def __init__(self, width): self.width, self.calls = width, []
            def __call__(self, ids, **kwargs):
                self.calls.append(ids.data)
                hidden = Tensor(np.repeat(ids.data[:, :, None], self.width, axis=2))
                pooled = Tensor([[len(self.calls)] * self.width])
                class Output:
                    hidden_states = [None, hidden, None]
                    def __getitem__(self, index): return pooled
                return Output()

        first, second = Encoder(2), Encoder(3)
        pipe = types.SimpleNamespace(tokenizer=Tokenizer(), tokenizer_2=Tokenizer(),
            text_encoder=first, text_encoder_2=second, _execution_device='cpu')
        self.assertIsNone(studio.long_prompt_embeddings(pipe, 'short', '', torch))
        self.assertEqual(first.calls, [])
        result = studio.long_prompt_embeddings(pipe, ' '.join(['word'] * 76), 'negative', torch)
        for key in ['prompt_embeds', 'negative_prompt_embeds']:
            self.assertEqual(result[key].data.shape, (1, 154, 5))
        self.assertEqual(result['pooled_prompt_embeds'].data.shape, (1, 3))
        self.assertTrue(np.all(result['pooled_prompt_embeds'].data == 1.5))
        # The empty alignment block must not dilute negative pooled conditioning.
        self.assertTrue(np.all(result['negative_pooled_prompt_embeds'].data == 3))
        self.assertEqual(first.calls[0][0, 0], 1000)
        self.assertEqual(first.calls[0][0, -1], 1001)
        self.assertEqual(first.calls[1][0, 1], 76)
        self.assertEqual(first.calls[1][0, 2], 1001)
        self.assertTrue(np.all(first.calls[1][0, 3:] == 0))
        with self.assertRaises(ValueError):
            studio.long_prompt_embeddings(pipe, '', 'word ' * 201, torch)
