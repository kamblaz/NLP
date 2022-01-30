import { Component, OnInit } from '@angular/core';
import axios from 'axios';
import { DataService } from '../data.service';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import { Observable } from 'rxjs';

@Component({
  selector: 'app-review',
  templateUrl: './review.component.html',
  styleUrls: ['./review.component.css']
})
export class ReviewComponent implements OnInit {

  review = 'Example';

  constructor(private http: HttpClient) {
  }

  ngOnInit(): void {
    //this.dataService.sendGetRequest().subscribe((data: any[])=>{
      //console.log(data);
    //})  
  }

  //send(): Observable<any> {
  send(): void {
    let input = (document.getElementById("review") as HTMLInputElement).value
    let output = (document.getElementById("rate") as HTMLOutputElement)   
    axios.post('http://localhost:5000/getReview', {
      review: input,
    }).then(response => {
      console.log(response.data);
      output.innerHTML = response.data;
    });
    /*const headers = { 'content-type': 'application/json'}  
    const body=JSON.stringify(input);
    return this.http.post('http://localhost:5000/getReview', body,{'headers':headers})*/
  }
}
