import { useState, useCallback } from 'react';

interface ErrorState {
  hasError: boolean;
  error: string | null;
  errorCode?: string;
  timestamp?: Date;
}

interface UseErrorHandlerResult {
  errorState: ErrorState;
  setError: (error: string, code?: string) => void;
  clearError: () => void;
  handleAsync: <T>(
    asyncFn: () => Promise<T>,
    errorMessage?: string
  ) => Promise<T | null>;
}

export const useErrorHandler = (): UseErrorHandlerResult => {
  const [errorState, setErrorState] = useState<ErrorState>({
    hasError: false,
    error: null,
  });

  const setError = useCallback((error: string, code?: string) => {
    setErrorState({
      hasError: true,
      error,
      errorCode: code,
      timestamp: new Date(),
    });
  }, []);

  const clearError = useCallback(() => {
    setErrorState({
      hasError: false,
      error: null,
    });
  }, []);

  const handleAsync = useCallback(async <T>(
    asyncFn: () => Promise<T>,
    errorMessage = '작업 중 오류가 발생했습니다.'
  ): Promise<T | null> => {
    try {
      clearError();
      return await asyncFn();
    } catch (err) {
      const message = err instanceof Error ? err.message : errorMessage;
      setError(message);
      return null;
    }
  }, [setError, clearError]);

  return {
    errorState,
    setError,
    clearError,
    handleAsync,
  };
};