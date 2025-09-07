class RVQEncoder:
	"""Placeholder RVQ encoder (no real logic)."""

	def __init__(self, stages=2, codebook_sizes=(128, 64)):
		self.stages = stages
		self.codebook_sizes = codebook_sizes

	def forward(self, features):
		"""Return fake indices list and placeholder residuals."""
		indices = [0] * self.stages
		return indices, None 