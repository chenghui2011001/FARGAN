class FarGanDecoder:
	"""Placeholder FarGan-style subframe decoder wrapper (no real logic)."""

	def __init__(self, subframe_ms=2.5, glus=3):
		self.subframe_ms = subframe_ms
		self.glus = glus

	def forward(self, rvq_condition):
		return None 