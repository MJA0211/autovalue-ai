# Synthetic fixtures only

This directory may contain tiny, explicitly synthetic records used by automated
tests. Never copy rows from the real dataset into a committed fixture unless its
redistribution terms have been verified and attribution is preserved.

Vehicle fixtures represent an invented U.S. market and must use
`market_country="US"` and `currency="USD"`. They are never approved production
training data.
