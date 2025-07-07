import { Injectable } from '@angular/core';
import { HttpClient, HttpHeaders, HttpParams } from '@angular/common/http';
import { Observable, map } from 'rxjs';
import { finalize } from 'rxjs/operators';
import { ConfigService } from '../config/config.service';
@Injectable({
  providedIn: 'root'
})
export class ApiCallService {

  baseUrl: string = "http://127.0.0.1:5050";

  constructor(private configService: ConfigService,
    private httpClient: HttpClient
  ) {
      //this.baseUrl = this.configService.baseURL
      console.log(this.baseUrl, 'config url')
   }

   generateResponse(data: any): any {
    console.log(data, 'data')
    return this.httpClient.get(`${this.baseUrl}/search`, {
      params: data
    }).pipe(
      map(res => res)
    );
  }

  // recentJobs(): any {
  //   return this.httpClient.get(`${this.baseUrl}/recent/`).pipe(map((res => res)))
  // }

  // getJobs(data: any): any {
  //   const params = new HttpParams({ fromObject: data });
  //   return this.httpClient.get(`${this.baseUrl}/search/`, { params }).pipe(map((res) => res));
  // }

  // getJobDetails(id: string): any {
  //   return this.httpClient.get(`${this.baseUrl}/jobs/${id}`).pipe(map((res => res)))
  // }
}
