import { ComponentFixture, TestBed } from '@angular/core/testing';

import { VerificationResult } from './verification-result';

describe('VerificationResult', () => {
  let component: VerificationResult;
  let fixture: ComponentFixture<VerificationResult>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [VerificationResult],
    }).compileComponents();

    fixture = TestBed.createComponent(VerificationResult);
    component = fixture.componentInstance;
    await fixture.whenStable();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
