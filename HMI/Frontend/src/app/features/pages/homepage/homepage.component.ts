import { Component, OnInit, HostListener } from '@angular/core';
import { ApiCallService } from '../../../core/services/api/api-call.service';
import { HttpClient, HttpErrorResponse } from '@angular/common/http';
import { CommonModule } from '@angular/common';
import { SubSink } from 'subsink';
import { NgxPaginationModule } from 'ngx-pagination';
import { FormBuilder, ReactiveFormsModule } from '@angular/forms';

@Component({
  selector: 'app-homepage',
  standalone: true,
  imports: [CommonModule, NgxPaginationModule, ReactiveFormsModule],
  templateUrl: './homepage.component.html',
  styleUrls: ['./homepage.component.scss'],
  providers: [ApiCallService]
})
export class HomepageComponent implements OnInit {

  subs = new SubSink();
  myForm = this.fb.group({
    message: ['']
  })
  response: any = "";
  isGeneration: boolean = false;

  constructor(public apiService: ApiCallService, 
    private http: HttpClient,
    private fb: FormBuilder
  ) { }

  ngOnInit(): void {
  }

  submit(): void {
    this.isGeneration = true;
    this.response = "";
    let data = {
      query: this.myForm.value.message,
    }
    
    this.subs.sink = this.apiService.generateResponse(data).subscribe((res: any) => {
      this.response = res.response;
      this.isGeneration = false;
    })
  }
  ngOnDestroy(): void {
    this.subs.unsubscribe();
  }
}
