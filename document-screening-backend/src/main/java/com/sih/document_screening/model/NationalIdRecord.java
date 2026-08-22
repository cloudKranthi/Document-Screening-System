package com.sih.document_screening.model;
  
import jakarta.persistence.*;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;
import java.time.LocalDate;

@Entity
@Table(name = "national_id_records")
@Getter
@Setter
@NoArgsConstructor
public class NationalIdRecord {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @OneToOne
    @JoinColumn(name = "screening_id", nullable = false)
    private Screening screening;

    private String nationalIdNumber;
    private String fullName;
    private LocalDate dateOfBirth;
    
    private Double ocrConfidence;
}
