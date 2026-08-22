package com.sih.document_screening.model;

import jakarta.persistence.*;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;
import java.time.LocalDate;

@Entity
@Table(name = "visa_records")
@Getter
@Setter
@NoArgsConstructor
public class VisaRecord {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @OneToOne
    @JoinColumn(name = "screening_id", nullable = false)
    private Screening screening;

    private String visaNumber;
    private String visaType;
    private LocalDate validUntil;
    private Integer stayDurationDays;
    
    private Double ocrConfidence;
}