
import { Component, inject } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { HttpClient, HttpErrorResponse } from '@angular/common/http';
import { timeout, catchError } from 'rxjs/operators';
import { throwError } from 'rxjs';

interface ExtractedClaim {
  claim_id: string;
  claim: string;
}

interface GenerateResponse {
  gemini_response?: string;
  claims?: ExtractedClaim[];
  error?: string;
}

@Component({
  selector: 'app-dashboard',
  imports: [FormsModule],
  templateUrl: './dashboard.html',
  styleUrl: './dashboard.css'
})
export class Dashboard {
  private http = inject(HttpClient);

  question = '';
  showResult = false;
  isLoading = false;
  errorMessage = '';

  geminiResponse = '';
  claims: ExtractedClaim[] = [];

  evidence = 'Evidence retrieval not connected yet.';
  similarity = 0;
  nliLabel = 'Not evaluated';
  confidence = 0;
  finalResult = 'NOT VERIFIED YET';
  explanation = '';

  verifyResponse(): void {
    const prompt = this.question.trim();

    if (!prompt || this.isLoading) {
      return;
    }

    this.isLoading = true;
    this.errorMessage = '';
    this.showResult = false;

    this.http.post<GenerateResponse>(
      'http://127.0.0.1:5000/generate',
      { prompt }
    ).pipe(
      timeout(20000),
      catchError((error) => {
        if (error.name === 'TimeoutError') {
          return throwError(() =>
            new Error(
              'Gemini is taking too long. Please try again later.'
            )
          );
        }

        return throwError(() => error);
      })
    ).subscribe({
      next: (response) => {
        console.log('Backend response:', response);

        if (!response.gemini_response) {
          this.errorMessage =
            response.error || 'No response received from Gemini.';
          this.isLoading = false;
          return;
        }

        this.geminiResponse = response.gemini_response;
        this.claims = response.claims ?? [];

        this.evidence = 'Evidence retrieval not connected yet.';
        this.similarity = 0;
        this.nliLabel = 'Not evaluated';
        this.confidence = 0;
        this.finalResult = 'NOT VERIFIED YET';
        this.explanation =
          'Claims have been extracted, but evidence retrieval and NLI verification are not connected yet.';

        this.showResult = true;
        this.isLoading = false;
      },
      error: (error: HttpErrorResponse | Error) => {
        console.error('Backend request failed:', error);

        if (error.name === 'TimeoutError' ||
            error.message?.includes('Gemini is taking too long')) {
          this.errorMessage =
            'Gemini is taking too long. Please try again later.';
        } else if (error instanceof HttpErrorResponse) {
          this.errorMessage =
            error.error?.error ||
            'Could not connect to the backend. Check that Flask is running.';
        } else {
          this.errorMessage =
            error.message || 'An unexpected error occurred.';
        }

        this.isLoading = false;
      }
    });
  }
}
