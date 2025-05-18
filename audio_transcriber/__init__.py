"""Package initialization for audio transcriber modules."""
from .tdt_asr import AudioTranscriber as TDTTranscriber
from .ctc_asr import AudioTranscriber as CTCTranscriber

__all__ = ['TDTTranscriber', 'CTCTranscriber']
